use anyhow::{Result, anyhow};
use ms_core::{SimulationBox, Trajectory, ThermoData};
use crate::{LoaderConfig, TrajectoryBuilder, ParsedAtom, Engine, FileFormat, DynamicParser};

pub struct LammpsEngine;

impl Engine for LammpsEngine {
    fn identify(&self, _filename: &str, bytes: &[u8]) -> Option<FileFormat> {
        let head = String::from_utf8_lossy(&bytes[..2048.min(bytes.len())]);
        let lines: Vec<&str> = head.lines().take(100).collect();
        if lines.iter().any(|l| l.contains("ITEM: TIMESTEP")) { return Some(FileFormat::LammpsDump); }
        if lines.iter().any(|l| {
            l.contains("Loop time of") || l.contains("LAMMPS (") || l.contains("Performance:") || l.contains("MPI processor grid")
        }) { return Some(FileFormat::LammpsLog); }
        if lines.iter().any(|l| {
            let t = l.trim();
            (t.contains("xlo xhi") || t.ends_with(" atoms")) && t.chars().next().map_or(false, |c| c.is_ascii_digit())
        }) { return Some(FileFormat::LammpsData); }
        if lines.iter().any(|l| {
            let t = l.trim(); t.starts_with("fix ") || t.starts_with("pair_style") || t.starts_with("units ")
        }) { return Some(FileFormat::LammpsIn); }
        None
    }

    fn create_parser(
        &self,
        files: &[(String, Vec<u8>)],
        config: &LoaderConfig,
        preferred_format: Option<FileFormat>,
        primary_filename: Option<&str>
    ) -> Result<Option<DynamicParser>> {
        if let Some(pref) = preferred_format {
            if !matches!(pref, FileFormat::LammpsDump | FileFormat::LammpsData | FileFormat::LammpsLog | FileFormat::LammpsIn) { return Ok(None); }
        }

        let mut current_config = config.clone();
        let mut identified_files = Vec::new();
        let mut has_primary = false;
        for (name, bytes) in files {
            let fmt = self.identify(name, bytes);
            if fmt.is_some() { has_primary = true; }
            identified_files.push((name, bytes, fmt));
        }
        if !has_primary { return Ok(None); }

        let mut main_builder = TrajectoryBuilder::new(&current_config);
        let mut data_file_found = false;

        // 1. Scan Data files FIRST to build element mapping
        for (_name, bytes, fmt) in &identified_files {
            if *fmt == Some(FileFormat::LammpsData) {
                use crate::lammps_data::{LammpsDataIncrementalParser, LammpsAtomStyle};
                let mut parser = LammpsDataIncrementalParser::new((*bytes).clone(), None, &current_config, LammpsAtomStyle::Auto)?;
                let _ = parser.parse_chunk(usize::MAX);
                let mut data_builder = parser.into_builder();
                
                // Finalize the mapping from data file
                let mut mapping = std::collections::HashMap::new();
                for (type_id, element) in &data_builder.config.element_mapping {
                    current_config.element_mapping.insert(*type_id, element.clone());
                    mapping.insert(*type_id, (element.clone(), 1.0));
                }
                current_config.time_step = data_builder.config.time_step;
                current_config.time_unit = data_builder.config.time_unit;
                
                // IMPORTANT: update atoms ALREADY added from data file
                data_builder.update_atom_types(&mapping);
                data_builder.config = current_config.clone();
                main_builder = data_builder;
                data_file_found = true;
                break;
            }
        }

        // 2. Return Dump parser with the updated config
        let mut dump_file = None;
        if let Some(primary) = primary_filename {
            for (name, bytes, fmt) in &identified_files {
                if *name == primary && *fmt == Some(FileFormat::LammpsDump) { dump_file = Some(bytes); break; }
            }
        }
        if dump_file.is_none() {
            for (_name, bytes, fmt) in &identified_files {
                if *fmt == Some(FileFormat::LammpsDump) { dump_file = Some(bytes); break; }
            }
        }

        if let Some(bytes) = dump_file {
            let mut parser = LammpsIncrementalParser::new((*bytes).clone(), None, &current_config)?;
            parser.builder = main_builder; 
            parser.builder.config = current_config;
            return Ok(Some(DynamicParser::LammpsDump(parser)));
        }

        if data_file_found {
            return Ok(Some(DynamicParser::LammpsData(crate::lammps_data::LammpsDataIncrementalParser::from_builder(main_builder))));
        }
        Ok(Some(DynamicParser::Empty(current_config)))
    }

    fn parse_thermo(&self, bytes: &[u8]) -> Result<ThermoData> { parse_lammps_log(bytes) }
}

pub fn parse_lammps_log(bytes: &[u8]) -> Result<ThermoData> {
    let content = std::str::from_utf8(bytes)?;
    let mut fields = Vec::new(); let mut thermo = ThermoData::default(); let mut in_thermo = false;
    for line in content.lines() {
        let line = line.trim(); if line.is_empty() { continue; }
        if line.starts_with("Step") {
            fields = line.split_whitespace().map(|s| s.to_string()).collect();
            thermo.fields = fields.clone(); in_thermo = true; continue;
        }
        if in_thermo {
            if line.starts_with("Loop time of") || line.starts_with("ERROR") || line.starts_with("WARNING") { in_thermo = false; continue; }
            if !line.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '-' || c == '.') {
                if line.contains("---") { continue; }
                if !thermo.timesteps.is_empty() { in_thermo = false; }
                continue;
            }
            let values: Vec<f64> = line.split_whitespace().filter_map(|s| fast_float::parse(s).ok()).collect();
            if values.len() == fields.len() && !values.is_empty() { let step = values[0] as u64; thermo.add_row(step, values); }
        }
    }
    if thermo.timesteps.is_empty() { return Err(anyhow!("No thermodynamic data found in log file")); }
    Ok(thermo)
}

pub struct LammpsIncrementalParser {
    bytes: Vec<u8>, pub(crate) builder: TrajectoryBuilder, offset: usize, current_step: u64, current_box: SimulationBox, col_map: Option<AtomColumnMap>, num_atoms: usize, frame_buffer: Vec<[f32; 3]>, temp_atoms: Vec<ParsedAtom>, state: ParserState, is_done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CoordinateType { Scaled, Unscaled }

struct AtomColumnMap {
    id: usize, type_id: usize, x: usize, y: usize, z: usize, coord_type: CoordinateType, max_idx: usize,
}

enum ParserState { Header, Atoms { remaining: usize, first_frame: bool }, }

impl LammpsIncrementalParser {
    pub fn new(bytes: Vec<u8>, _data_bytes: Option<Vec<u8>>, config: &LoaderConfig) -> Result<Self> {
        Ok(Self { bytes, builder: TrajectoryBuilder::new(config), offset: 0, current_step: 0, current_box: SimulationBox::default(), col_map: None, num_atoms: 0, frame_buffer: Vec::new(), temp_atoms: Vec::new(), state: ParserState::Header, is_done: false, })
    }
    pub fn is_done(&self) -> bool { self.is_done } 
    pub fn offset(&self) -> usize { self.offset }
    pub fn progress(&self) -> f32 { if self.bytes.is_empty() { 1.0 } else { (self.offset as f32 / self.bytes.len() as f32).min(1.0) } }
    pub fn status_message(&self) -> String { format!("Reading LAMMPS dump... {:.1} MB", self.offset as f32 / 1_048_576.0) }
    pub fn parse_chunk(&mut self, max_lines: usize) -> Result<()> {
        let mut lines_done = 0; let bytes_len = self.bytes.len();
        while lines_done < max_lines && self.offset < bytes_len {
            match self.state {
                ParserState::Header => {
                    let (s, e) = self.next_line_range(); let line = std::str::from_utf8(&self.bytes[s..e])?.trim(); lines_done += 1; if line.is_empty() { continue; }
                    if line.starts_with("ITEM: TIMESTEP") {
                        if !self.frame_buffer.is_empty() { let pos = std::mem::replace(&mut self.frame_buffer, vec![[0.0; 3]; self.num_atoms]); self.builder.add_frame_data(pos, self.current_step, self.current_box, None)?; } else { self.builder.clear_frames(); }
                        let (s, e) = self.next_line_range(); self.current_step = std::str::from_utf8(&self.bytes[s..e])?.trim().parse().unwrap_or(0);
                    } else if line.starts_with("ITEM: NUMBER OF ATOMS") {
                        let (s, e) = self.next_line_range(); self.num_atoms = std::str::from_utf8(&self.bytes[s..e])?.trim().parse().unwrap_or(0);
                        if self.frame_buffer.is_empty() { self.frame_buffer = vec![[0.0; 3]; self.num_atoms]; let est_frames = self.bytes.len() / (self.num_atoms * 45 + 200).max(1); self.builder.reserve(est_frames, self.num_atoms); }
                    } else if line.starts_with("ITEM: BOX BOUNDS") {
                        let (s1, e1) = self.next_line_range(); let (s2, e2) = self.next_line_range(); let (s3, e3) = self.next_line_range();
                        self.current_box = parse_box_str(std::str::from_utf8(&self.bytes[s1..e1])?.trim(), std::str::from_utf8(&self.bytes[s2..e2])?.trim(), std::str::from_utf8(&self.bytes[s3..e3])?.trim()).unwrap_or_default();
                    } else if line.starts_with("ITEM: ATOMS") {
                        if self.col_map.is_none() { if let Ok(m) = AtomColumnMap::from_header(line) { self.col_map = Some(m); } }
                        let is_first = self.builder.num_atoms().is_none(); self.state = ParserState::Atoms { remaining: self.num_atoms, first_frame: is_first };
                    }
                }
                ParserState::Atoms { remaining, first_frame } => {
                    let (s, e) = self.next_line_range(); let line = std::str::from_utf8(&self.bytes[s..e])?.trim(); lines_done += 1; if line.is_empty() { continue; }
                    if let Some(ref map) = self.col_map {
                        if let Ok(atom) = map.parse_line(line) {
                            if first_frame {
                                self.temp_atoms.push(atom);
                                if self.temp_atoms.len() == self.num_atoms {
                                    let atoms = std::mem::take(&mut self.temp_atoms); self.builder.set_initial_atoms(atoms.clone());
                                    for a in atoms { if let Some(idx) = self.builder.get_index_for_id(a.id) { let mut p = a.pos; if map.coord_type == CoordinateType::Scaled { p = unscale_coords(p, &self.current_box); } self.frame_buffer[idx] = p; } }
                                }
                            } else if let Some(idx) = self.builder.get_index_for_id(atom.id) { let mut p = atom.pos; if map.coord_type == CoordinateType::Scaled { p = unscale_coords(p, &self.current_box); } self.frame_buffer[idx] = p; }
                        }
                    }
                    let new_remaining = remaining - 1;
                    if new_remaining == 0 { self.state = ParserState::Header; } else { self.state = ParserState::Atoms { remaining: new_remaining, first_frame }; }
                }
            }
        }
        if self.offset >= bytes_len { if !self.frame_buffer.is_empty() { let pos = std::mem::take(&mut self.frame_buffer); self.builder.add_frame_data(pos, self.current_step, self.current_box, None)?; } self.is_done = true; }
        Ok(())
    }
    fn next_line_range(&mut self) -> (usize, usize) { let remaining = &self.bytes[self.offset..]; let line_len = remaining.iter().position(|&b| b == b'\n').map(|p| p + 1).unwrap_or(remaining.len()); let start = self.offset; let end = start + line_len; self.offset = end; (start, end) }
    pub fn finish(self) -> Result<Trajectory> { self.builder.finish() }
}

impl AtomColumnMap {
    fn from_header(header_line: &str) -> Result<Self> {
        let parts: Vec<&str> = header_line.split_whitespace().collect();
        let find_idx = |names: &[&str]| -> Result<usize> {
            for name in names { if let Some(pos) = parts.iter().position(|&s| s == *name) { return Ok(pos - 2); } } 
            Err(anyhow!("Required column not found"))
        };
        let x_idx = find_idx(&["x", "xs", "xu", "xsu"])?; let id = find_idx(&["id"])?; let type_id = find_idx(&["type"])?; let y = find_idx(&["y", "ys", "yu", "ysu"])?; let z = find_idx(&["z", "zs", "zu", "zsu"])?;
        let coord_name = parts.get(x_idx + 2).ok_or_else(|| anyhow!("Invalid header"))?;
        let coord_type = if coord_name.contains('s') { CoordinateType::Scaled } else { CoordinateType::Unscaled };
        Ok(Self { id, type_id, x: x_idx, y, z, coord_type, max_idx: id.max(type_id).max(x_idx).max(y).max(z) })
    }
    fn parse_line(&self, line: &str) -> Result<ParsedAtom> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() <= self.max_idx { return Err(anyhow!("Line too short")); }
        let id = parts[self.id].parse()?;
        let type_id = parts[self.type_id].parse()?;
        let x = fast_float::parse(parts[self.x])?;
        let y = fast_float::parse(parts[self.y])?;
        let z = fast_float::parse(parts[self.z])?;
        Ok(ParsedAtom { id, type_id, pos: [x, y, z] })
    }
}

fn unscale_coords(pos: [f32; 3], b: &SimulationBox) -> [f32; 3] {
    let [xs, ys, zs] = pos;
    let z = b.z_range[0] + zs * (b.z_range[1] - b.z_range[0]);
    let y = b.y_range[0] + ys * (b.y_range[1] - b.y_range[0]) + zs * b.yz;
    let x = b.x_range[0] + xs * (b.x_range[1] - b.x_range[0]) + ys * b.xy + zs * b.xz;
    [x, y, z]
}

fn parse_box_str(x: &str, y: &str, z: &str) -> Result<SimulationBox> {
    let p = |s: &str| -> Result<(f32, f32, f32)> {
        let v: Vec<&str> = s.split_whitespace().collect();
        if v.len() < 2 { return Err(anyhow!("Invalid box bounds")); }
        let lo = fast_float::parse(v[0])?; let hi = fast_float::parse(v[1])?;
        let tilt = if v.len() > 2 { fast_float::parse(v[2])? } else { 0.0 };
        Ok((lo, hi, tilt))
    };
    let (xlo, xhi, xy) = p(x)?; let (ylo, yhi, xz) = p(y)?; let (zlo, zhi, yz) = p(z)?;
    Ok(SimulationBox { x_range: [xlo, xhi], y_range: [ylo, yhi], z_range: [zlo, zhi], xy, xz, yz })
}

#[derive(Debug, Default, Clone)]
pub struct LammpsMetadata {
    pub timestep: Option<f32>,
    pub time_unit: Option<ms_core::TimeUnit>,
    pub mass_map: std::collections::HashMap<u32, f32>,
    pub bonds_found: bool,
}

pub fn extract_lammps_metadata(bytes: &[u8]) -> LammpsMetadata {
    let mut meta = LammpsMetadata::default();
    let content = String::from_utf8_lossy(&bytes[..bytes.len().min(100_000)]);
    let mut in_masses = false;
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("timestep") || line.starts_with("fix") && line.contains("dt") {
            if let Some(val) = line.split_whitespace().last().and_then(|s| s.parse::<f32>().ok()) { meta.timestep = Some(val); }
        } else if line.starts_with("units") {
            if line.contains("real") { meta.time_unit = Some(ms_core::TimeUnit::Femtoseconds); }
            else if line.contains("metal") { meta.time_unit = Some(ms_core::TimeUnit::Picoseconds); }
        } else if line.contains("Masses") { in_masses = true; }
        else if line.contains("Atoms") || line.contains("Bonds") { 
            in_masses = false; 
            if line.contains("Bonds") { meta.bonds_found = true; } 
        }
        else if line.contains("bonds") || line.contains("bond types") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if !parts.is_empty() {
                if let Ok(n) = parts[0].parse::<usize>() {
                    if n > 0 { meta.bonds_found = true; }
                }
            }
        }
        else if in_masses {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 { if let (Ok(id), Ok(m)) = (parts[0].parse::<u32>(), parts[1].parse::<f32>()) { meta.mass_map.insert(id, m); } }
        }
    }
    meta
}
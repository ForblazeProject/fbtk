use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct AtomInfo {
    pub index: usize,
    pub element: String,
    pub resname: String,
    pub atom_type: usize,
}

pub struct SelectionEngine {
    atoms: Vec<AtomInfo>,
}

impl SelectionEngine {
    pub fn new(atoms: Vec<AtomInfo>) -> Self {
        Self { atoms }
    }

    /// Select atoms based on a query string.
    pub fn select(&self, query: &str) -> Result<Vec<usize>> {
        let query = query.trim().to_lowercase();
        
        // Handle "and" by splitting and intersecting
        if query.contains(" and ") {
            let parts: Vec<&str> = query.split(" and ").collect();
            let mut result = self.select(parts[0])?;
            for part in &parts[1..] {
                let set: std::collections::HashSet<_> = self.select(part)?.into_iter().collect();
                result.retain(|x| set.contains(x));
            }
            return Ok(result);
        }

        // 1. Simple element/type selection
        if query.len() <= 2 && query.chars().all(|c| c.is_alphabetic()) {
            return Ok(self.atoms.iter()
                .filter(|a| a.element.eq_ignore_ascii_case(&query))
                .map(|a| a.index)
                .collect());
        }

        let parts: Vec<&str> = query.split_whitespace().collect();
        if parts.is_empty() { return Ok(Vec::new()); }

        match parts[0] {
            "element" => {
                if parts.len() < 2 { return Err(anyhow!("Missing element name")); }
                let target = parts[1];
                Ok(self.atoms.iter()
                    .filter(|a| a.element.eq_ignore_ascii_case(target))
                    .map(|a| a.index)
                    .collect())
            }
            "type" => {
                if parts.len() < 2 { return Err(anyhow!("Missing type ID")); }
                let target_type: usize = parts[1].parse()?;
                Ok(self.atoms.iter()
                    .filter(|a| a.atom_type == target_type)
                    .map(|a| a.index)
                    .collect())
            }
            "resname" | "residue" => {
                if parts.len() < 2 { return Err(anyhow!("Missing residue name")); }
                let target = parts[1];
                Ok(self.atoms.iter()
                    .filter(|a| a.resname.eq_ignore_ascii_case(target))
                    .map(|a| a.index)
                    .collect())
            }
            "index" => {
                if parts.len() < 2 { return Err(anyhow!("Missing index range")); }
                if query.contains(" to ") {
                    let start: usize = parts[1].parse()?;
                    let end: usize = parts[3].parse()?;
                    Ok((start..=end).collect())
                } else {
                    let range_parts: Vec<&str> = parts[1].split(':').collect();
                    if range_parts.len() == 2 {
                        let start: usize = range_parts[0].parse()?;
                        let end: usize = range_parts[1].parse()?;
                        Ok((start..end).collect())
                    } else {
                        let mut res = Vec::new();
                        for p in &parts[1..] {
                            if let Ok(idx) = p.parse::<usize>() { res.push(idx); }
                        }
                        Ok(res)
                    }
                }
            }
            "not" => {
                let sub = self.select(&query[4..])?;
                let set: std::collections::HashSet<_> = sub.into_iter().collect();
                Ok(self.atoms.iter()
                    .map(|a| a.index)
                    .filter(|i| !set.contains(i))
                    .collect())
            }
            _ => {
                // Default to element if just a word
                Ok(self.atoms.iter()
                    .filter(|a| a.element.eq_ignore_ascii_case(parts[0]))
                    .map(|a| a.index)
                    .collect())
            }
        }
    }

    /// Parse a pair query for RDF (e.g., "O-H" or "A with B")
    pub fn select_pair(&self, query: &str) -> Result<(Vec<usize>, Vec<usize>)> {
        let query_lower = query.to_lowercase();
        let (sep, split_at) = if let Some(pos) = query_lower.find(" with ") {
            (" with ", pos)
        } else if let Some(pos) = query_lower.find(" - ") {
            (" - ", pos)
        } else if let Some(pos) = query_lower.find("-") {
            ("-", pos)
        } else {
            let idx = self.select(query)?;
            return Ok((idx.clone(), idx));
        };

        let idx_a = self.select(&query[..split_at])?;
        let idx_b = self.select(&query[split_at + sep.len()..])?;
        Ok((idx_a, idx_b))
    }
}

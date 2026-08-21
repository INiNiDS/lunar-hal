pub mod env;
pub mod time;

pub fn encode_rgb_png(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;

    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let i = (y * w + x) * 3;
            raw.push(rgb[i]);
            raw.push(rgb[i + 1]);
            raw.push(rgb[i + 2]);
        }
    }

    let deflate = deflate_minimal(&raw);

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    let ihdr_data = ihdr_chunk(width, height);
    png.extend_from_slice(&ihdr_data);

    let idat_data = idat_chunk(&deflate);
    png.extend_from_slice(&idat_data);

    let iend = [0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82];
    png.extend_from_slice(&iend);

    png
}

pub fn ihdr_chunk(width: u32, height: u32) -> [u8; 25] {
    let mut out = [0u8; 25];
    out[0..4].copy_from_slice(&13u32.to_be_bytes());
    out[4..8].copy_from_slice(b"IHDR");
    out[8..12].copy_from_slice(&width.to_be_bytes());
    out[12..16].copy_from_slice(&height.to_be_bytes());
    out[16] = 8;
    out[17] = 2;
    out[18..21].copy_from_slice(&[0, 0, 0]);
    let crc = crc32(&out[4..21]);
    out[21..25].copy_from_slice(&crc.to_be_bytes());
    out
}

pub fn idat_chunk(deflated: &[u8]) -> Vec<u8> {
    let len = deflated.len() as u32;
    let mut out = Vec::with_capacity(12 + deflated.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(b"IDAT");
    out.extend_from_slice(deflated);
    let crc = crc32(&out[4..4 + 4 + deflated.len()]);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

pub fn deflate_minimal(data: &[u8]) -> Vec<u8> {
    let max_block = 65535;
    let num_blocks = data.len().div_ceil(max_block);
    let mut compressed = Vec::with_capacity(data.len() + num_blocks * 5 + 6);
    compressed.push(0x78);
    compressed.push(0x01);

    let mut offset = 0;
    for i in 0..num_blocks {
        let end = (offset + max_block).min(data.len());
        let block_len = end - offset;
        let bfinal: u8 = if i == num_blocks - 1 { 1 } else { 0 };
        compressed.push(bfinal);
        compressed.extend_from_slice(&(block_len as u16).to_le_bytes());
        compressed.extend_from_slice(&(!(block_len as u16)).to_le_bytes());
        compressed.extend_from_slice(&data[offset..end]);
        offset = end;
    }

    compressed.extend_from_slice(&adler32(data).to_be_bytes());
    compressed
}

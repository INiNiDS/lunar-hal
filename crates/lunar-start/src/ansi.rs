
fn skip_csi_sequence<I>(chars: &mut std::iter::Peekable<I>)
where
    I: Iterator<Item = char>,
{
    if let Some('[') = chars.peek() {
        chars.next();
        while let Some(&next) = chars.peek() {
            chars.next();
            if ('@'..='~').contains(&next) {
                break;
            }
        }
    }
}

pub fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            skip_csi_sequence(&mut chars);
        } else {
            result.push(c);
        }
    }
    result
}

pub fn clean_line(s: &str) -> String {
    strip_ansi(s).replace('\r', "")
}
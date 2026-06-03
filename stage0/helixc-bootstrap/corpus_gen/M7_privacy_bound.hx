// M-7 (charter HELIX_COMPLETION.md §1.6 MED) -- DOCUMENT-AS-BOUND + negative privacy test.
// BOUND: the bootstrap has NO module privacy enforcement. `pub` is parsed as a
// no-op (parser.hx:1798); `mod name { ... }` function-mangles inner fns
// (hidden -> secret__hidden, parser.hx:parse_mod_decl) and they are reachable
// from anywhere via the `secret::hidden()` path REGARDLESS of `pub`. A full
// visibility/privacy pass + filesystem module loader is v-next (only if the
// user asks for cross-module privacy). See §1.6 M-7 / spec module section.
//
// NEGATIVE PRIVACY TEST: call a NON-`pub` (private) module fn across the module
// boundary. In Rust this is a COMPILE ERROR ("function `hidden` is private").
// The bootstrap ACCEPTS it (no privacy check) and runs the body -> 42. Exit 42
// therefore PROVES the absence of privacy enforcement (a private item is freely
// callable). If a visibility pass is later added (reject the private call), this
// row fails and flags that the documented bound has changed (intended).
mod secret {
    fn hidden() -> i32 { 42 }   // NOT pub -- private in Rust; here freely callable
}
fn main() -> i32 {
    secret::hidden()             // Rust: error[E0603] private; bootstrap: 42
}

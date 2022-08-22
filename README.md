# fused_error

A simple library for working with composable errors.

[![Build Status]][actions]
[![Latest Version]][crates.io]

[Build Status]: https://img.shields.io/github/workflow/status/seancroach/fused_error/ci?logo=github
[actions]: https://github.com/seancroach/fused_error/actions/workflows/ci.yml
[Latest Version]: https://img.shields.io/crates/v/fused_errorlogo=rust
[crates.io]: https://crates.io/crates/fused_error

## Documentation

[Module documentation with examples](https://docs.rs/fused_error). The module documentation also
includes a comprehensive description of the syntax supported for parsing hex colors.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
fused_error = "0.1.0"
```

Here's a simple example to demonstrate the power of composable errors:

```rust
use std::{
    num::{IntErrorKind, ParseIntError},
    iter::Sum,
    str::FromStr
};

use fused_error::{Accumulator, FusedResult, IteratorExt};

/// Take an iterator of textual data, adding up all of the parsed numbers.
///
/// Unlike the standard way of returning a `Result<N, N::Err>`, this doesn't
/// short-circuit, it keeps track of the current sum, and reports any
/// further diagnostics past the first failure.
fn calculate_sum<N, E, I>(iter: I) -> FusedResult<N, N::Err>
where
    N: FromStr + Sum,
    E: AsRef<str>,
    I: IntoIterator<Item = E>,
{
    // Error accumulators collect errors to defer handling them, providing
    // more holistic diagnostics:
    let mut acc = Accumulator::new();
    let sum = iter
        .into_iter()
        .map(|item| item.as_ref().parse::<N>())
        // `fused_error` adds certain methods to iterators; no more
        // disrupting iterator chains and `collect` hells for results!
        .accumulate(&mut acc)
        .sum();
    // fused results let you easily pass around error accumulators and
    // are perfect for cases where a yielded "ok" value and an error case
    // aren't mutually exclusive.
    FusedResult::new(sum, acc)
}

let result: FusedResult<i32, _> = calculate_sum(["1", "2", "3", "4"]);
assert_eq!(result.value(), &10);
assert_eq!(result.errors(), []);

let result: FusedResult<i8, _> = calculate_sum(["", "-129", "foo", "128"]);
assert_eq!(result.value(), &0);
assert_eq!(
    result
        .errors()
        .into_iter()
        .map(|err| err.kind().clone())
        .collect::<Vec<_>>(),
    [
        IntErrorKind::Empty,
        IntErrorKind::NegOverflow,
        IntErrorKind::InvalidDigit,
        IntErrorKind::PosOverflow,
    ],
);

let result: FusedResult<u8, _> = calculate_sum(["-1", "", "0", "1"]);
assert_eq!(result.value(), &1);
assert_eq!(
    result
        .errors()
        .into_iter()
        .map(|err| err.kind().clone())
        .collect::<Vec<_>>(),
    [IntErrorKind::InvalidDigit, IntErrorKind::Empty],
);
```

Or, when using the `syn` feature for increased interoperability, here's an
example of `fused_error` assisting in procedural macros:

```rust
use fused_error::{Accumulator, FusedError};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use syn::{AttributeArgs, DeriveInput, ItemFn};

#[proc_macro_attribute]
pub fn my_attribute(args: TokenStream, input: TokenStream) -> TokenStream {
    fn inner(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream2> {
        let mut acc = Accumulator::<syn::Error>::new();

        // fn(TokenStream) -> syn::Result<AttributeArgs>
        let args = acc.handle(parse_args(args));
        let item = acc.handle(syn::parse::<ItemFn>(input));

        // Throw all of the collected parse errors, or continue:
        acc = acc.checkpoint()?;

        // SAFETY: any `None` value would short-circuit at the last
        // checkpoint.
        let mut args = unsafe { args.unwrap_unchecked() };
        let mut item = unsafe { item.unwrap_unchecked() };

        // fn(&AttributeArgs) -> syn::Result<()>
        acc.handle(validate_args(&args));
        // fn(&ItemFn) -> syn::Result<()>
        acc.handle(validate_item(&item));

        // Throw all of the validation parse errors, or continue:
        acc = acc.checkpoint()?;

        // Do multiple steps that can short-circuit:
        let tokens = acc.handle_in(|| {
            // fn(&mut AttributeArgs, &mut ItemFn) -> syn::Result<()>
            prepare(&mut args, &mut item)?;
            // fn(AttributeArgs, ItemFn) -> syn::Result<TokenStream2>
            expand(args, item)
        });

        // If this closure is called, we know `tokens` is `Some`:
        acc.err_or_else(|| unsafe { tokens.unwrap_unchecked() })
    }

    inner(args, input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
```

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the
work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

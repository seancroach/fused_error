//! Extending iterator functionality with [`IteratorExt`].

use crate::{accumulator::Accumulator, IntoResultParts};

use std::iter::FusedIterator;

/// An iterator that takes an input of `Result<T, IE>` items and accumulates the
/// errors into an accumulator of error type `E`, yielding an output stream of
/// `T` items. Errors of `IE` must implement [`Into<E>`].
///
/// This `struct` is created by the [`accumulate`](IteratorExt::accumulate)
/// method on [`Iterator`] items via [`IteratorExt`].
pub struct Accumulate<'a, I, E> {
    iter: I,
    acc: &'a mut Accumulator<E>,
}

impl<'a, I, E> Accumulate<'a, I, E> {
    fn new(iter: I, acc: &'a mut Accumulator<E>) -> Self {
        Accumulate { iter, acc }
    }
}

impl<'a, I, E> Iterator for Accumulate<'a, I, E>
where
    I: Iterator,
    I::Item: IntoResultParts,
    <I::Item as IntoResultParts>::Err: Into<E>,
{
    type Item = <I::Item as IntoResultParts>::Ok;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.find_map(|res| self.acc.handle(res))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, I, E> ExactSizeIterator for Accumulate<'a, I, E>
where
    Self: Iterator,
    I: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, I, E> FusedIterator for Accumulate<'a, I, E>
where
    Self: Iterator,
    I: FusedIterator,
{
}

/// Extends [`Iterator`] with methods for complex error handling:
pub trait IteratorExt: Iterator {
    /// Creates an iterator that filters results, collecting errors into an
    /// [error accumulator](Accumulator) and yielding an iterator over all of
    /// the "ok" values.
    ///
    /// `accumulate` can be used to make chains of [`filter`], [`map`], and
    /// [`handle`] more concise. The example below shows how a common
    /// [`filter_map`] call can be shortened.
    ///
    /// [`filter`]: Iterator::filter
    /// [`map`]: Iterator::map
    /// [`handle`]: Accumulator::handle
    /// [`filter_map`]: Iterator::filter_map
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use fused_error::{Accumulator, IteratorExt};
    ///
    /// let results = [
    ///     Ok(1),
    ///     Err("foo"),
    ///     Ok(3),
    ///     Err("bar"),
    ///     Ok(5),
    /// ];
    ///
    /// let mut acc = Accumulator::<&str>::new();
    ///
    /// let sum: i32 = results
    ///     .into_iter()
    ///     .accumulate(&mut acc)
    ///     .sum();
    ///
    /// assert_eq!(sum, 9);
    /// assert_eq!(acc.into_vec(), ["foo", "bar"]);
    /// ```
    ///
    /// Here's the same example, but with [`filter_map`]:
    ///
    /// ```
    /// use fused_error::Accumulator;
    ///
    /// let results = [
    ///     Ok(1),
    ///     Err("foo"),
    ///     Ok(3),
    ///     Err("bar"),
    ///     Ok(5),
    /// ];
    ///
    /// let mut acc = Accumulator::<&str>::new();
    ///
    /// let sum: i32 = results
    ///     .into_iter()
    ///     .filter_map(|res| acc.handle(res))
    ///     .sum();
    ///
    /// assert_eq!(sum, 9);
    /// assert_eq!(acc.into_vec(), ["foo", "bar"]);
    /// ```
    #[inline]
    fn accumulate<E>(self, acc: &mut Accumulator<E>) -> Accumulate<Self, E>
    where
        Self: Sized,
        Self::Item: IntoResultParts,
        <Self::Item as IntoResultParts>::Err: Into<E>,
    {
        Accumulate::new(self, acc)
    }

    /// Drains the errors from an iterator of results into an
    /// [error accumulator](Accumulator), discarding any "ok" values.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, IteratorExt};
    ///
    /// let results = [Err("foo"), Ok(()), Err("bar")];
    /// let mut accumulator = Accumulator::<&str>::new();
    ///
    /// results.into_iter().collect_errors(&mut accumulator);
    ///
    /// assert_eq!(accumulator.into_vec(), ["foo", "bar"]);
    /// ```
    #[inline]
    fn collect_errors<E>(self, acc: &mut Accumulator<E>)
    where
        Self: Sized,
        Self::Item: IntoResultParts,
        <Self::Item as IntoResultParts>::Err: Into<E>,
    {
        self.for_each(|res| {
            let (_, err) = res.into_result_parts();
            acc.extend(err);
        });
    }

    /// Unwraps an iterator of results, collecting all errors into a new
    /// [error accumulator](Accumulator) and "ok" values into some collection
    /// that implements [`FromIterator`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::{Accumulator, IteratorExt};
    ///
    /// let results = [
    ///     Ok("foo"),
    ///     Err("bar"),
    ///     Ok("baz"),
    ///     Err("qux"),
    /// ];
    ///
    /// let (vec, acc): (Vec<_>, _) = results.into_iter().partition_results();
    /// assert_eq!(vec, ["foo", "baz"]);
    /// assert_eq!(acc.into_vec(), ["bar", "qux"]);
    /// ```
    fn partition_results<C>(self) -> (C, Accumulator<<Self::Item as IntoResultParts>::Err>)
    where
        Self: Sized,
        Self::Item: IntoResultParts,
        C: FromIterator<<Self::Item as IntoResultParts>::Ok>,
    {
        let mut acc = Accumulator::new();
        let c: C = self.accumulate(&mut acc).collect::<C>();
        (c, acc)
    }
}

impl<I> IteratorExt for I where I: Iterator {}

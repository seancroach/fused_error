//! Working with remote error types using [`Accumulated<E>`].

use crate::FusedError;

use std::{
    borrow::{Borrow, BorrowMut},
    fmt::{self, Debug, Display},
    ops::{Deref, DerefMut, Index, IndexMut},
    slice::SliceIndex,
};

/// A thin wrapper to handle remote error types as if they implemented
/// [`FusedError`].
///
/// # Examples
///
/// ```
/// use fused_error::{Accumulated, Accumulator};
///
/// // `Accumulated<E>` is needed since Rust's "orphan rule" won't allow you to
/// // implement `FusedError` on `String`:
/// type Error = Accumulated<String>;
///
/// let mut acc = Accumulator::<Error>::new();
///
/// // A downside from using `Accumulated<E>` is that methods that would
/// // implicitly perform conversions before only accept `E`, `Vec<E>`,
/// // and `Accumulated<E>`:
/// acc.push("foo".to_string());
/// acc.push("bar".to_string());
///
/// assert_eq!(acc.finish().unwrap_err(), ["foo", "bar"]);
/// ```
#[derive(Clone, Eq, PartialOrd, Ord, Hash)]
pub struct Accumulated<E> {
    inner: Vec<E>,
}

impl<E> Accumulated<E> {
    /// Constructs a new, empty `Accumulated<E>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulated;
    ///
    /// let mut err = Accumulated::new();
    /// # err.push("foo");
    /// ```
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Accumulated::from_vec(Vec::new())
    }

    /// Constructs a new `Accumulated<E>` from a vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulated;
    ///
    /// let mut err = Accumulated::from_vec(vec!["foo", "bar"]);
    /// ```
    #[must_use]
    #[inline]
    pub fn from_vec(vec: Vec<E>) -> Self {
        Accumulated { inner: vec }
    }

    /// Unwraps the `Accumulated<E>` into the underlying vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use fused_error::Accumulated;
    ///
    /// let mut err = Accumulated::new();
    /// err.push("foo");
    /// err.push("bar");
    ///
    /// assert_eq!(err.into_vec(), ["foo", "bar"]);
    /// ```
    #[must_use]
    #[inline]
    pub fn into_vec(self) -> Vec<E> {
        self.inner
    }
}

impl<E> FusedError for Accumulated<E> {
    #[inline]
    fn combine(&mut self, mut other: Self) {
        self.inner.append(&mut other.inner);
    }
}

impl<E> Default for Accumulated<E> {
    #[must_use]
    #[inline]
    fn default() -> Self {
        Accumulated::new()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Deref traits
////////////////////////////////////////////////////////////////////////////////

impl<E> Deref for Accumulated<E> {
    type Target = Vec<E>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<E> DerefMut for Accumulated<E> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

////////////////////////////////////////////////////////////////////////////////
// Error traits
////////////////////////////////////////////////////////////////////////////////

impl<E> Debug for Accumulated<E>
where
    E: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

impl<E> Display for Accumulated<E>
where
    E: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let messages = self.iter().map(ToString::to_string).collect::<Vec<_>>();
        write!(
            f,
            "accumulated collection of {} errors: {messages:#?}",
            self.inner.len()
        )
    }
}

impl<E> std::error::Error for Accumulated<E> where E: Debug + Display {}

////////////////////////////////////////////////////////////////////////////////
// Conversion traits
////////////////////////////////////////////////////////////////////////////////

impl<E> From<Vec<E>> for Accumulated<E> {
    #[inline]
    fn from(vec: Vec<E>) -> Self {
        Accumulated::from_vec(vec)
    }
}

impl<E> From<Accumulated<E>> for Vec<E> {
    #[inline]
    fn from(acc: Accumulated<E>) -> Self {
        acc.into_vec()
    }
}

impl<E> From<E> for Accumulated<E> {
    #[inline]
    fn from(err: E) -> Self {
        Accumulated::from_vec(vec![err])
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterator traits
////////////////////////////////////////////////////////////////////////////////

impl<E> FromIterator<E> for Accumulated<E> {
    #[inline]
    fn from_iter<T: IntoIterator<Item = E>>(iter: T) -> Self {
        let vec = Vec::from_iter(iter);
        Accumulated::from_vec(vec)
    }
}

impl<E> Extend<E> for Accumulated<E> {
    #[inline]
    fn extend<T: IntoIterator<Item = E>>(&mut self, iter: T) {
        self.inner.extend(iter);
    }
}

impl<'a, E> IntoIterator for &'a Accumulated<E> {
    type Item = &'a E;
    type IntoIter = Iter<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, E> IntoIterator for &'a mut Accumulated<E> {
    type Item = &'a mut E;
    type IntoIter = IterMut<'a, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<E> IntoIterator for Accumulated<E> {
    type Item = E;
    type IntoIter = IntoIter<E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterator types
////////////////////////////////////////////////////////////////////////////////

/// Immutable `Accumulated<E>` iterator.
pub type Iter<'a, E> = std::slice::Iter<'a, E>;

/// Mutable `Accumulated<E>` iterator.
pub type IterMut<'a, E> = std::slice::IterMut<'a, E>;

/// An iterator that moves out of an `Accumulated<E>`.
pub type IntoIter<E> = std::vec::IntoIter<E>;

////////////////////////////////////////////////////////////////////////////////
// As traits
////////////////////////////////////////////////////////////////////////////////

impl<E> AsRef<[E]> for Accumulated<E> {
    #[inline]
    fn as_ref(&self) -> &[E] {
        self.inner.as_ref()
    }
}

impl<E> AsMut<[E]> for Accumulated<E> {
    #[inline]
    fn as_mut(&mut self) -> &mut [E] {
        self.inner.as_mut()
    }
}

impl<E> Borrow<[E]> for Accumulated<E> {
    #[inline]
    fn borrow(&self) -> &[E] {
        self.inner.deref().borrow()
    }
}

impl<E> BorrowMut<[E]> for Accumulated<E> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [E] {
        self.inner.deref_mut().borrow_mut()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Index traits
////////////////////////////////////////////////////////////////////////////////

impl<E, I> Index<I> for Accumulated<E>
where
    I: SliceIndex<[E]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        self.inner.deref().index(index)
    }
}

impl<E, I> IndexMut<I> for Accumulated<E>
where
    I: SliceIndex<[E]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.inner.deref_mut().index_mut(index)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Comparison traits
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_slice_eq {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty) => {
        impl<E, F, $($vars)*> PartialEq<$rhs> for $lhs
        where
            E: PartialEq<F>
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool {
                self[..] == other[..]
            }
        }
    }
}

impl_slice_eq!([] Accumulated<E>, Accumulated<F>);
impl_slice_eq!([] Accumulated<E>, Vec<F>);
impl_slice_eq!([] Vec<E>, Accumulated<F>);
impl_slice_eq!([] Accumulated<E>, [F]);
impl_slice_eq!([][E], Accumulated<F>);
impl_slice_eq!([] Accumulated<E>, &[F]);
impl_slice_eq!([] & [E], Accumulated<F>);
impl_slice_eq!([const N: usize] Accumulated<E>, [F; N]);
impl_slice_eq!([const N: usize] [E; N], Accumulated<F>);
impl_slice_eq!([const N: usize] Accumulated<E>, &[F; N]);
impl_slice_eq!([const N: usize] &[E; N], Accumulated<F>);

impl<E> PartialOrd<Vec<E>> for Accumulated<E>
where
    E: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Vec<E>) -> Option<std::cmp::Ordering> {
        self.inner.partial_cmp(other)
    }
}

impl<E> PartialOrd<Accumulated<E>> for Vec<E>
where
    E: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &Accumulated<E>) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.inner)
    }
}

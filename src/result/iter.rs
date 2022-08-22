use std::iter::FusedIterator;

/// An iterator over a reference to the "ok" value in a [`CompoundResult`].
///
/// This struct is created by the [`CompoundResult::iter`] method.
///
/// [`CompoundResult`]: crate::CompoundResult
/// [`CompoundResult::iter`]: crate::CompoundResult::iter
pub struct Iter<'a, T> {
    pub(crate) inner: Option<&'a T>,
}

macro_rules! iter_items {
    () => {
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.inner.take()
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let n = if self.inner.is_some() { 1 } else { 0 };
            (n, Some(n))
        }
    };
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    iter_items!();
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.take()
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {}

impl<T> FusedIterator for Iter<'_, T> {}

impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { inner: self.inner }
    }
}

/// An iterator over a mutable reference to the "ok" value in a
/// [`CompoundResult`].
///
/// This struct is created by the [`CompoundResult::iter_mut`] method.
///
/// [`CompoundResult`]: crate::CompoundResult
/// [`CompoundResult::iter_mut`]: crate::CompoundResult::iter_mut
pub struct IterMut<'a, T> {
    pub(crate) inner: Option<&'a mut T>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    iter_items!();
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.take()
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {}

impl<T> FusedIterator for IterMut<'_, T> {}

/// An iterator over  the "ok" value in a [`CompoundResult`].
///
/// This struct is created by the [`CompoundResult::into_iter`] method.
///
/// [`CompoundResult`]: crate::CompoundResult
/// [`CompoundResult::into_iter`]: crate::CompoundResult::into_iter
pub type IntoIter<T> = std::result::IntoIter<T>;

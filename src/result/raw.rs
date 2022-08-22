use crate::{Accumulator, FusedError};

use std::{
    fmt::{self, Debug},
    mem::ManuallyDrop,
};

#[must_use]
pub(crate) struct FusedResult<T, E> {
    pub ok: ManuallyDrop<T>,
    pub errors: Accumulator<E>,
}

impl<T, E> FusedResult<T, E> {
    #[inline]
    pub fn new(value: T, acc: Accumulator<E>) -> Self {
        FusedResult {
            ok: ManuallyDrop::new(value),
            errors: acc,
        }
    }

    #[inline]
    pub unsafe fn split(&mut self) -> (T, Accumulator<E>) {
        let value = ManuallyDrop::take(&mut self.ok);
        let acc = Accumulator::from_vec(self.errors.inner.take());
        (value, acc)
    }

    #[must_use]
    #[inline]
    pub unsafe fn take_value(&mut self) -> T {
        self.errors.inner.ignore();
        ManuallyDrop::take(&mut self.ok)
    }

    #[must_use]
    #[inline]
    pub unsafe fn take_errors(&mut self) -> Vec<E> {
        ManuallyDrop::drop(&mut self.ok);
        self.errors.inner.take()
    }

    #[must_use]
    #[inline]
    pub unsafe fn take_err(&mut self) -> Option<E>
    where
        E: FusedError,
    {
        ManuallyDrop::drop(&mut self.ok);
        self.errors.inner.reduce()
    }

    #[inline]
    pub unsafe fn result(&mut self) -> Result<T, E>
    where
        E: FusedError,
    {
        match self.errors.inner.reduce() {
            Some(err) => {
                ManuallyDrop::drop(&mut self.ok);
                Err(err)
            }
            None => {
                let ok = ManuallyDrop::take(&mut self.ok);
                Ok(ok)
            }
        }
    }

    #[inline]
    #[track_caller]
    pub unsafe fn fail_with(&mut self, msg: &str)
    where
        T: Debug,
        E: Debug,
    {
        let msg = format!("{msg}: {self:#?}");
        self.ignore();
        panic!("{msg}");
    }

    #[inline(never)]
    #[cold]
    #[track_caller]
    pub unsafe fn fail(&mut self, method: &'static str)
    where
        T: Debug,
        E: Debug,
    {
        let msg = match self.errors.len() {
            0 => format!("called `FusedResult::{method}` when no errors were present"),
            n => {
                format!(
                    "called `FusedResult::{method}` without handling {n} error{}",
                    if n == 1 { "" } else { "s" },
                )
            }
        };

        self.fail_with(&msg);
    }

    #[inline]
    pub unsafe fn ignore(&mut self) {
        ManuallyDrop::drop(&mut self.ok);
        self.errors.inner.ignore();
    }
}

// We manually implement `Debug` to omit the `ManuallyDrop` indirection.
impl<T: Debug, E: Debug> Debug for FusedResult<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedResult")
            .field("ok", &*self.ok)
            .field("errors", &self.errors)
            .finish()
    }
}

impl<T, E> Drop for FusedResult<T, E> {
    fn drop(&mut self) {
        if cfg!(debug_assertions) && !self.errors.is_handled() {
            // SAFETY: If the accumulator didn't get handled correctly, it's
            // safe to assume the `ManuallyDrop` hasn't been handled either.
            unsafe { ManuallyDrop::drop(&mut self.ok) };

            let len = self.errors.len();
            match len {
                0 => panic!("`fused_error::FusedResult` dropped without getting handled"),
                n => panic!(
                    "`fused_error::FusedResult` dropped without handling {n} error{}",
                    if n == 1 { "" } else { "s" },
                ),
            }
        }
    }
}

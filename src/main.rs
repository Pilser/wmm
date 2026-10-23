use std::process::ExitCode;

fn main() -> ExitCode {
    // Match normal Unix CLI behavior: exit quietly when the reader closes the
    // pipe (e.g. `wmm stats ... | head`), instead of panicking on EPIPE.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    match wmm::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

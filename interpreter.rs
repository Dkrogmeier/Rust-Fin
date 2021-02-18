use crate::lex::lex;
use crate::parse::parse;
use crate::eval::{EvalResult};

/// Lexes, parses, and evaluates the given program.
pub fn run_interpreter(program: &str) -> EvalResult {
    //EvalResult::Unit
    unimplemented!();
    /*
    match lex(program) {
        Err(err) => EvalResult::Err("Lex Error".into()),
        Ok(tokens) => match parse(&tokens) {
            Err(err) => EvalResult::Err("Parse Error".into()),
            Ok(expr) => {
                let mut env = Environment::default();
                match eval(expr.clone(), &mut env) {
                    EvalResult::Err(err) => EvalResult::Err(err),
                    _ => {}
                }
            },
        }
    }*/
}

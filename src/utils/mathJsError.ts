export class MathJsError extends Error {
    constructor(signature: string, message: string) {
        super(`MathJsError: Position=${signature}. message=${message}.`);
    }
}

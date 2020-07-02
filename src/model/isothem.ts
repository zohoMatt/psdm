import { Unit, concat, evaluate, multiply, divide, unit } from 'mathjs';
import {MathJsError} from "../utils/mathJsError";

export class Isothem {
    public static langumuir(concentration: Unit | Unit[], k: Unit, n: Unit): Unit | Unit[] {
        // (k * c * n) / (1 + k * c)
        const langmuirFn = (c: Unit) => {
            const sign = `fn@langmuir`;
            try {
                return evaluate(`
                    (${k}) * (${c}) * (${n}) / (1 + (${k}) * (${c}))
                `);
            } catch (e) {
                throw new MathJsError(sign, e.toString());
            }
        };

        if (concentration instanceof Array) {
            return concentration.map(langmuirFn);
        } else return langmuirFn(concentration);
    }


    public static freundlich() {
        // todo
    }

    public static redlichPeterson() {
        // todo
    }
}

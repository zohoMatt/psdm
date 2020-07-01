import { evaluate, unit, Unit } from 'mathjs';

export class Water {
    public static density(temperature: Unit | string): Unit {
        const numericT = unit(temperature.toString()).toNumber('degC');
        const kgpm3 = evaluate(
            `-0.000000136017092*${numericT}^4 + 0.000042945572506*${numericT}^3 - 0.007624720184594*${numericT}^2 + 0.054223297911449*${numericT} + 999.85329403558`
        );
        return unit(kgpm3, 'kg/m^3');
    }

    public static dynamicViscosity(t: Unit | string): Unit {
        const numericT = unit(t.toString()).toNumber('degC');
        const kv = unit(
            evaluate(
                `0.00000000000014005*${numericT}^4 - 0.000000000020716*${numericT}^3 + 0.0000000013967*${numericT}^2 - 0.000000059867*${numericT} + 0.0000017852`
            ),
            'm^2/s'
        );
        const density = Water.density(t);
        return unit(evaluate(`${density}*${kv}`));
    }
}

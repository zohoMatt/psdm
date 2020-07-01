import { evaluate, unit, Unit } from 'mathjs';

export class Bed {
    public static crossSectionalArea(diameter: Unit | string): Unit {
        return unit(evaluate(`pi * (${diameter.toString()})^2 / 4`));
    }

    public static volume(length: Unit | string, diameter: Unit | string): Unit {
        const area = Bed.crossSectionalArea(diameter).toString();
        return unit(evaluate(`(${area}) * ${length}`));
    }

    public static density(
        mass: Unit | string,
        length: Unit | string,
        diameter: Unit | string
    ): Unit {
        const volume = Bed.volume(length, diameter).toString();
        const m = mass.toString();
        return unit(evaluate(`(${m})/(${volume})`));
    }
}

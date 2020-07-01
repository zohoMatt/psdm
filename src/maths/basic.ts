export function interpolate(array: [number, number], delta: number) {
    return (array[1] - array[0]) * delta + array[0];
}

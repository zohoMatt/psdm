"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
exports.__esModule = true;
exports.MathJsError = void 0;
var MathJsError = /** @class */ (function (_super) {
    __extends(MathJsError, _super);
    function MathJsError(signature, message) {
        return _super.call(this, "MathJsError: Position=" + signature + ". message=" + message + ".") || this;
    }
    return MathJsError;
}(Error));
exports.MathJsError = MathJsError;

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = () => {
	return {
		target: ['web'],
		entry: './index.js',
		// entry: path.resolve(__dirname, 'index.js'),
		output: {
			path: path.resolve(__dirname, 'dist'),
			filename: 'bundle.min.js',
			library: {
				type: 'umd'
			}
		},
		// plugins: [new CopyPlugin({
		// 	patterns: [{ from: 'node_modules/onnxruntime-web/dist/ort-training-wasm-simd.wasm', to: '[name][ext]'}, 
		// 	{ from: 'node_modules/@xenova/transformers/dist/ort-training-wasm-simd.wasm', to: '[name][ext]'}]
		// })],
		mode: 'production',
		resolve: {
			extensions: ['.ts', '.js'],
		}
	}
};

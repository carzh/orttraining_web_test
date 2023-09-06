const ort = require('onnxruntime-web/training');

const chkptPath = 'assets/artifacts/mobilevit/checkpoint.ckpt';
const trainingPath = 'assets/artifacts/mobilevit/training_model.onnx';
const optimizerPath = 'assets/artifacts/mobilevit/optimizer_model.onnx';
const evalPath = 'assets/artifacts/mobilevit/eval_model.onnx';

async function main() {
	try {
		const checkpointState = await ort.TrainingSession.create(chkptPath, trainingPath, evalPath, optimizerPath);
		document.write('loading success!!!! whoohooooo~~');
	} catch (e) {
		document.write(`loading checkpoint failed: ${e}.`);
		document.write(':(');
		document.write(e.stack);
	}
}

main();

const ort = require('onnxruntime-web');

async function main() {
	try {
		const checkpointState = await ort.CheckpointState.loadCheckpoint('./assets/artifacts/paramtrain_tensors.pbseq');
		document.write('loading success!!!! whoohooooo~~');
	} catch (e) {
		document.write(`loading checkpoint failed: ${e}.`);
	}
}

main();

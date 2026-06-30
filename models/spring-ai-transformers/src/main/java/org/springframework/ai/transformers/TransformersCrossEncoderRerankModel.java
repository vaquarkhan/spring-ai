/*
 * Copyright 2023-present the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.transformers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.Document;
import org.springframework.ai.rerank.RerankModel;
import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;
import org.springframework.ai.rerank.RerankResult;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

/**
 * An ONNX-based cross-encoder reranking model implementation. Cross-encoders take a
 * (query, passage) pair as input and output a relevance score, making them ideal for
 * reranking retrieved documents.
 *
 * <p>
 * By default, it uses the cross-encoder/ms-marco-MiniLM-L-6-v2 model, but can be
 * configured to use any ONNX-compatible cross-encoder model. The class supports both CPU
 * and GPU inference, caching of model resources, and various tokenization options.
 * </p>
 *
 * <p>
 * For cross-encoder models, see:
 * <a href="https://www.sbert.net/docs/pretrained-models/ce-msmarco.html">SBERT
 * Cross-Encoders</a>
 * </p>
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public class TransformersCrossEncoderRerankModel implements RerankModel, InitializingBean {

	private static final Log logger = LogFactory.getLog(TransformersCrossEncoderRerankModel.class);

	/**
	 * Default ONNX tokenizer URI for cross-encoder/ms-marco-MiniLM-L-6-v2.
	 */
	public static final String DEFAULT_ONNX_TOKENIZER_URI = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json";

	/**
	 * Default ONNX model URI for cross-encoder/ms-marco-MiniLM-L-6-v2.
	 */
	public static final String DEFAULT_ONNX_MODEL_URI = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx";

	private Resource tokenizerResource = toResource(DEFAULT_ONNX_TOKENIZER_URI);

	private Resource modelResource = toResource(DEFAULT_ONNX_MODEL_URI);

	private Map<String, String> tokenizerOptions = Map.of();

	private int gpuDeviceId = -1;

	private @Nullable String resourceCacheDirectory;

	private boolean disableCaching = false;

	@SuppressWarnings("NullAway.Init")
	private HuggingFaceTokenizer tokenizer;

	private final OrtEnvironment environment = OrtEnvironment.getEnvironment();

	@SuppressWarnings("NullAway.Init")
	private OrtSession session;

	@SuppressWarnings("NullAway.Init")
	private Set<String> onnxModelInputs;

	@SuppressWarnings("NullAway.Init")
	private ResourceCacheService cacheService;

	public TransformersCrossEncoderRerankModel() {
	}

	public void setTokenizerResource(Resource tokenizerResource) {
		this.tokenizerResource = tokenizerResource;
	}

	public void setTokenizerResource(String tokenizerResourceUri) {
		this.tokenizerResource = toResource(tokenizerResourceUri);
	}

	public void setModelResource(Resource modelResource) {
		this.modelResource = modelResource;
	}

	public void setModelResource(String modelResourceUri) {
		this.modelResource = toResource(modelResourceUri);
	}

	public void setTokenizerOptions(Map<String, String> tokenizerOptions) {
		this.tokenizerOptions = tokenizerOptions;
	}

	public void setGpuDeviceId(int gpuDeviceId) {
		this.gpuDeviceId = gpuDeviceId;
	}

	public void setResourceCacheDirectory(@Nullable String resourceCacheDirectory) {
		this.resourceCacheDirectory = resourceCacheDirectory;
	}

	public void setDisableCaching(boolean disableCaching) {
		this.disableCaching = disableCaching;
	}

	@Override
	public void afterPropertiesSet() throws Exception {
		this.cacheService = StringUtils.hasText(this.resourceCacheDirectory)
				? new ResourceCacheService(this.resourceCacheDirectory) : new ResourceCacheService();

		this.tokenizer = HuggingFaceTokenizer.newInstance(getCachedResource(this.tokenizerResource).getInputStream(),
				this.tokenizerOptions);

		try (var sessionOptions = new OrtSession.SessionOptions()) {
			if (this.gpuDeviceId >= 0) {
				sessionOptions.addCUDA(this.gpuDeviceId);
			}
			this.session = this.environment.createSession(getCachedResource(this.modelResource).getContentAsByteArray(),
					sessionOptions);
		}

		this.onnxModelInputs = this.session.getInputNames();
		Set<String> onnxModelOutputs = this.session.getOutputNames();

		logger.info(
				"Cross-encoder model input names: " + this.onnxModelInputs.stream().collect(Collectors.joining(", ")));
		logger.info("Cross-encoder model output names: " + onnxModelOutputs.stream().collect(Collectors.joining(", ")));
	}

	@Override
	public RerankResponse call(RerankRequest request) {
		Assert.notNull(request, "request must not be null");
		List<Document> documents = request.getDocuments();
		if (documents.isEmpty()) {
			return new RerankResponse(List.of());
		}

		String query = request.getQuery();
		List<RerankResult> results = new ArrayList<>();

		try {
			// Cross-encoder requires (query, passage) pairs. Encode each pair separately.
			for (int i = 0; i < documents.size(); i++) {
				String passage = documents.get(i).getText();
				double score = computeRelevanceScore(query, passage != null ? passage : "");
				results.add(new RerankResult(documents.get(i), score, i));
			}
		}
		catch (OrtException ex) {
			throw new RuntimeException("Failed to run cross-encoder inference", ex);
		}

		return new RerankResponse(results);
	}

	private double computeRelevanceScore(String query, String passage) throws OrtException {
		// Cross-encoders use paired input: tokenize (query, passage) together
		Encoding encoding = this.tokenizer.encode(query, passage);

		long[][] inputIds = new long[][] { encoding.getIds() };
		long[][] attentionMask = new long[][] { encoding.getAttentionMask() };
		long[][] tokenTypeIds = new long[][] { encoding.getTypeIds() };

		try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(this.environment, inputIds);
				OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(this.environment, attentionMask);
				OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(this.environment, tokenTypeIds)) {

			Map<String, OnnxTensor> modelInputs = Map.of("input_ids", inputIdsTensor, "attention_mask",
					attentionMaskTensor, "token_type_ids", tokenTypeIdsTensor);

			modelInputs = removeUnknownModelInputs(modelInputs);

			try (OrtSession.Result onnxResults = this.session.run(modelInputs)) {
				// Cross-encoder output is a logit of shape [batch_size, 1] or
				// [batch_size, num_labels]
				OnnxValue output = onnxResults.get(0);
				float[][] logits = (float[][]) output.getValue();
				// For single-label regression cross-encoders, the score is logits[0][0]
				return logits[0][0];
			}
		}
	}

	private Map<String, OnnxTensor> removeUnknownModelInputs(Map<String, OnnxTensor> modelInputs) {
		return modelInputs.entrySet()
			.stream()
			.filter(entry -> this.onnxModelInputs.contains(entry.getKey()))
			.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
	}

	private Resource getCachedResource(Resource resource) {
		return this.disableCaching ? resource : this.cacheService.getCachedResource(resource);
	}

	private static Resource toResource(String uri) {
		return new DefaultResourceLoader().getResource(uri);
	}

}

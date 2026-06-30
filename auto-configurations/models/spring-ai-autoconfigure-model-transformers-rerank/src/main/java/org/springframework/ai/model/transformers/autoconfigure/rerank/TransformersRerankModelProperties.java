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

package org.springframework.ai.model.transformers.autoconfigure.rerank;

import java.util.HashMap;
import java.util.Map;

import org.springframework.ai.transformers.TransformersCrossEncoderRerankModel;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * Configuration properties for the Transformer Cross-Encoder Reranking model.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
@ConfigurationProperties(TransformersRerankModelProperties.CONFIG_PREFIX)
public class TransformersRerankModelProperties {

	public static final String CONFIG_PREFIX = "spring.ai.rerank.transformer";

	@NestedConfigurationProperty
	private final Tokenizer tokenizer = new Tokenizer();

	@NestedConfigurationProperty
	private final Cache cache = new Cache();

	@NestedConfigurationProperty
	private final Onnx onnx = new Onnx();

	public Cache getCache() {
		return this.cache;
	}

	public Onnx getOnnx() {
		return this.onnx;
	}

	public Tokenizer getTokenizer() {
		return this.tokenizer;
	}

	public static class Tokenizer {

		private String uri = TransformersCrossEncoderRerankModel.DEFAULT_ONNX_TOKENIZER_URI;

		private Map<String, String> options = new HashMap<>();

		public String getUri() {
			return this.uri;
		}

		public void setUri(String uri) {
			this.uri = uri;
		}

		public Map<String, String> getOptions() {
			return this.options;
		}

		public void setOptions(Map<String, String> options) {
			this.options = options;
		}

	}

	public static class Cache {

		private boolean enabled = true;

		private String directory = "";

		public boolean isEnabled() {
			return this.enabled;
		}

		public void setEnabled(boolean enabled) {
			this.enabled = enabled;
		}

		public String getDirectory() {
			return this.directory;
		}

		public void setDirectory(String directory) {
			this.directory = directory;
		}

	}

	public static class Onnx {

		private String modelUri = TransformersCrossEncoderRerankModel.DEFAULT_ONNX_MODEL_URI;

		private int gpuDeviceId = -1;

		public String getModelUri() {
			return this.modelUri;
		}

		public void setModelUri(String modelUri) {
			this.modelUri = modelUri;
		}

		public int getGpuDeviceId() {
			return this.gpuDeviceId;
		}

		public void setGpuDeviceId(int gpuDeviceId) {
			this.gpuDeviceId = gpuDeviceId;
		}

	}

}

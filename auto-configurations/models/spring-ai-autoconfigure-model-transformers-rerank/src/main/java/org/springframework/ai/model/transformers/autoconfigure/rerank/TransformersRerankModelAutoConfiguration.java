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

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OrtSession;

import org.springframework.ai.rerank.RerankModel;
import org.springframework.ai.transformers.TransformersCrossEncoderRerankModel;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * {@link AutoConfiguration Auto-configuration} for the ONNX Transformers Cross-Encoder
 * Reranking Model.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
@AutoConfiguration
@EnableConfigurationProperties(TransformersRerankModelProperties.class)
@ConditionalOnClass({ OrtSession.class, HuggingFaceTokenizer.class, TransformersCrossEncoderRerankModel.class })
public class TransformersRerankModelAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(RerankModel.class)
	public TransformersCrossEncoderRerankModel rerankModel(TransformersRerankModelProperties properties) {
		TransformersCrossEncoderRerankModel rerankModel = new TransformersCrossEncoderRerankModel();

		rerankModel.setDisableCaching(!properties.getCache().isEnabled());
		if (!properties.getCache().getDirectory().isEmpty()) {
			rerankModel.setResourceCacheDirectory(properties.getCache().getDirectory());
		}

		rerankModel.setTokenizerResource(properties.getTokenizer().getUri());
		rerankModel.setTokenizerOptions(properties.getTokenizer().getOptions());

		rerankModel.setModelResource(properties.getOnnx().getModelUri());
		rerankModel.setGpuDeviceId(properties.getOnnx().getGpuDeviceId());

		return rerankModel;
	}

}

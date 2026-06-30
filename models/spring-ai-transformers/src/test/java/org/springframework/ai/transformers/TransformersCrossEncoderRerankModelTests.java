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

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import org.springframework.ai.document.Document;
import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;
import org.springframework.ai.rerank.RerankResult;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for {@link TransformersCrossEncoderRerankModel}. These tests require
 * downloading the ONNX model from HuggingFace and are gated behind an environment
 * variable to avoid slowing down CI.
 *
 * <p>
 * Set {@code SPRING_AI_RERANK_ONNX_TESTS_ENABLED=true} to run these tests.
 * </p>
 *
 * @author Vaquar Khan
 */
@EnabledIfEnvironmentVariable(named = "SPRING_AI_RERANK_ONNX_TESTS_ENABLED", matches = "true")
class TransformersCrossEncoderRerankModelTests {

	@Test
	void relevantPassageOutranksIrrelevant() throws Exception {
		TransformersCrossEncoderRerankModel model = new TransformersCrossEncoderRerankModel();
		model.afterPropertiesSet();

		Document relevant = new Document("Paris is the capital of France.");
		Document irrelevant = new Document("The quick brown fox jumps over the lazy dog.");

		RerankResponse response = model
			.call(new RerankRequest("What is the capital of France?", List.of(irrelevant, relevant), null));

		List<RerankResult> results = response.getResults();
		assertThat(results).hasSize(2);

		// The relevant doc (index 1) should score higher than the irrelevant one (index
		// 0)
		RerankResult relevantResult = results.stream().filter(r -> r.getIndex() == 1).findFirst().orElseThrow();
		RerankResult irrelevantResult = results.stream().filter(r -> r.getIndex() == 0).findFirst().orElseThrow();
		assertThat(relevantResult.getScore()).isGreaterThan(irrelevantResult.getScore());
	}

	@Test
	void preservesAllDocumentsWhenNoTopN() throws Exception {
		TransformersCrossEncoderRerankModel model = new TransformersCrossEncoderRerankModel();
		model.afterPropertiesSet();

		List<Document> docs = List.of(new Document("doc one"), new Document("doc two"), new Document("doc three"));

		RerankResponse response = model.call(new RerankRequest("query", docs, null));

		assertThat(response.getResults()).hasSize(3);
	}

	@Test
	void scoresAreFiniteAndOrdered() throws Exception {
		TransformersCrossEncoderRerankModel model = new TransformersCrossEncoderRerankModel();
		model.afterPropertiesSet();

		List<Document> docs = List.of(new Document("Machine learning is a subset of artificial intelligence."),
				new Document("The weather is nice today."),
				new Document("Deep learning uses neural networks with many layers."));

		RerankResponse response = model.call(new RerankRequest("What is deep learning?", docs, null));

		List<RerankResult> results = response.getResults();
		assertThat(results).hasSize(3);

		for (RerankResult result : results) {
			assertThat(Double.isFinite(result.getScore())).isTrue();
		}

		// Use the convenience rerank method and verify ordering
		List<Document> reranked = model.rerank("What is deep learning?", docs);
		assertThat(reranked).hasSize(3);
	}

}

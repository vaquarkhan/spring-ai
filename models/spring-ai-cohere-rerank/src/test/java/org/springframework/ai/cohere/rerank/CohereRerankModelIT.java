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

package org.springframework.ai.cohere.rerank;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import org.springframework.ai.document.Document;
import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;
import org.springframework.ai.rerank.RerankResult;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for {@link CohereRerankModel}. Requires a valid Cohere API key in the
 * {@code COHERE_API_KEY} environment variable.
 *
 * @author Vaquar Khan
 */
@EnabledIfEnvironmentVariable(named = "COHERE_API_KEY", matches = ".+")
class CohereRerankModelIT {

	private final CohereRerankModel model = new CohereRerankModel(System.getenv("COHERE_API_KEY"));

	@Test
	void relevantPassageOutranksIrrelevant() {
		Document relevant = new Document("Paris is the capital of France.");
		Document irrelevant = new Document("The quick brown fox jumps over the lazy dog.");

		RerankResponse response = this.model
			.call(new RerankRequest("What is the capital of France?", List.of(irrelevant, relevant), null));

		List<RerankResult> results = response.getResults();
		assertThat(results).hasSize(2);

		RerankResult relevantResult = results.stream().filter(r -> r.getIndex() == 1).findFirst().orElseThrow();
		RerankResult irrelevantResult = results.stream().filter(r -> r.getIndex() == 0).findFirst().orElseThrow();
		assertThat(relevantResult.getScore()).isGreaterThan(irrelevantResult.getScore());
	}

	@Test
	void respectsTopN() {
		List<Document> docs = List.of(new Document("Paris is the capital of France."),
				new Document("Berlin is the capital of Germany."),
				new Document("The quick brown fox jumps over the lazy dog."));

		CohereRerankOptions options = CohereRerankOptions.builder().topN(2).build();
		RerankResponse response = this.model.call(new RerankRequest("European capitals", docs, options));

		assertThat(response.getResults()).hasSize(2);
	}

	@Test
	void returnsFiniteScores() {
		List<Document> docs = List.of(new Document("Machine learning is a subset of AI."),
				new Document("The weather is nice today."), new Document("Deep learning uses neural networks."));

		RerankResponse response = this.model.call(new RerankRequest("What is deep learning?", docs, null));

		assertThat(response.getResults()).hasSize(3);
		for (RerankResult result : response.getResults()) {
			assertThat(Double.isFinite(result.getScore())).isTrue();
			assertThat(result.getScore()).isBetween(0.0, 1.0);
		}
	}

}

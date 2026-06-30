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

import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for {@link CohereRerankModel} that do not require a Cohere API key.
 *
 * @author Vaquar Khan
 */
class CohereRerankModelTests {

	@Test
	void emptyInputReturnsEmptyResponse() {
		CohereRerankModel model = new CohereRerankModel("test-key");
		RerankResponse response = model.call(new RerankRequest("query", List.of(), null));
		assertThat(response.getResults()).isEmpty();
	}

	@Test
	void constructorRejectsBlankApiKey() {
		assertThatThrownBy(() -> new CohereRerankModel("")).isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new CohereRerankModel("   ")).isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void defaultOptionsUseExpectedModel() {
		CohereRerankOptions options = CohereRerankOptions.builder().model(CohereRerankOptions.DEFAULT_MODEL).build();
		assertThat(options.getModel()).isEqualTo("rerank-v3.5");
	}

	@Test
	void optionsBuilderSetsAllFields() {
		CohereRerankOptions options = CohereRerankOptions.builder()
			.model("rerank-english-v3.0")
			.topN(10)
			.scoreThreshold(0.5)
			.maxChunksPerDoc(5)
			.build();
		assertThat(options.getModel()).isEqualTo("rerank-english-v3.0");
		assertThat(options.getTopN()).isEqualTo(10);
		assertThat(options.getScoreThreshold()).isEqualTo(0.5);
		assertThat(options.getMaxChunksPerDoc()).isEqualTo(5);
	}

	@Test
	void callRejectsNullRequest() {
		CohereRerankModel model = new CohereRerankModel("test-key");
		assertThatThrownBy(() -> model.call(null)).isInstanceOf(IllegalArgumentException.class);
	}

}

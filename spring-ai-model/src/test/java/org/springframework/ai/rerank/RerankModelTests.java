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

package org.springframework.ai.rerank;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for the {@link RerankModel} abstraction and related types.
 *
 * @author Vaquar Khan
 */
class RerankModelTests {

	private final Map<String, Double> scoreMap = Map.of("low", 0.1, "mid", 0.5, "high", 0.9);

	private final RerankModel stubModel = new StubRerankModel(this.scoreMap);

	@Test
	void rerankReordersByDescendingScore() {
		List<Document> docs = List.of(doc("low"), doc("mid"), doc("high"));
		List<Document> result = this.stubModel.rerank("q", docs);
		assertThat(result).extracting(Document::getText).containsExactly("high", "mid", "low");
	}

	@Test
	void rerankReturnsEmptyForEmptyInput() {
		List<Document> result = this.stubModel.rerank("q", List.of());
		assertThat(result).isEmpty();
	}

	@Test
	void responseGetResultReturnsFirstAndNullWhenEmpty() {
		RerankResponse populated = new RerankResponse(
				List.of(new RerankResult(doc("a"), 0.8, 0), new RerankResult(doc("b"), 0.6, 1)));
		assertThat(populated.getResult()).isNotNull();
		assertThat(populated.getResult().getOutput().getText()).isEqualTo("a");

		RerankResponse empty = new RerankResponse(List.of());
		assertThat(empty.getResult()).isNull();
	}

	@Test
	void requestExposesQueryDocumentsAndOptions() {
		List<Document> docs = List.of(doc("x"), doc("y"));
		RerankOptions opts = DefaultRerankOptions.builder().model("test-model").topN(3).scoreThreshold(0.5).build();
		RerankRequest request = new RerankRequest("my query", docs, opts);

		assertThat(request.getQuery()).isEqualTo("my query");
		assertThat(request.getInstructions()).hasSize(2);
		assertThat(request.getRerankOptions()).isNotNull();
		assertThat(request.getRerankOptions().getModel()).isEqualTo("test-model");
		assertThat(request.getRerankOptions().getTopN()).isEqualTo(3);
		assertThat(request.getRerankOptions().getScoreThreshold()).isEqualTo(0.5);
	}

	@Test
	void requestRejectsBlankQuery() {
		assertThatThrownBy(() -> new RerankRequest("", List.of(doc("a")), null))
			.isInstanceOf(IllegalArgumentException.class);
		assertThatThrownBy(() -> new RerankRequest("   ", List.of(doc("a")), null))
			.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void optionsDefaultToNull() {
		RerankOptions opts = DefaultRerankOptions.builder().build();
		assertThat(opts.getModel()).isNull();
		assertThat(opts.getTopN()).isNull();
		assertThat(opts.getScoreThreshold()).isNull();
	}

	private static Document doc(String text) {
		return new Document(text);
	}

	/**
	 * A stub rerank model that scores documents based on a pre-configured map. Returns
	 * results in the ORIGINAL input order (unsorted) to prove that consumers must sort.
	 */
	static class StubRerankModel implements RerankModel {

		private final Map<String, Double> scores;

		StubRerankModel(Map<String, Double> scores) {
			this.scores = scores;
		}

		@Override
		public RerankResponse call(RerankRequest request) {
			List<Document> docs = request.getDocuments();
			List<RerankResult> results = IntStream.range(0, docs.size()).mapToObj(i -> {
				Document doc = docs.get(i);
				double score = this.scores.getOrDefault(doc.getText(), 0.0);
				return new RerankResult(doc, score, i);
			}).collect(Collectors.toList());
			return new RerankResponse(results);
		}

	}

}

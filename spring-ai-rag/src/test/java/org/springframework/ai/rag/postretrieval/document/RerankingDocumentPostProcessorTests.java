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

package org.springframework.ai.rag.postretrieval.document;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rerank.RerankModel;
import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;
import org.springframework.ai.rerank.RerankResult;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Unit tests for {@link RerankingDocumentPostProcessor}.
 *
 * @author Vaquar Khan
 */
class RerankingDocumentPostProcessorTests {

	private final Map<String, Double> scoreMap = Map.of("low", 0.1, "mid", 0.5, "high", 0.9);

	private final RerankModel stubModel = new StubRerankModel(this.scoreMap);

	@Test
	void reordersByDescendingScore() {
		RerankingDocumentPostProcessor processor = RerankingDocumentPostProcessor.builder()
			.rerankModel(this.stubModel)
			.build();

		List<Document> result = processor.process(new Query("q"), List.of(doc("low"), doc("high"), doc("mid")));

		assertThat(result).extracting(Document::getText).containsExactly("high", "mid", "low");
	}

	@Test
	void truncatesToTopN() {
		RerankingDocumentPostProcessor processor = RerankingDocumentPostProcessor.builder()
			.rerankModel(this.stubModel)
			.topN(2)
			.build();

		List<Document> result = processor.process(new Query("q"), List.of(doc("low"), doc("high"), doc("mid")));

		assertThat(result).extracting(Document::getText).containsExactly("high", "mid");
	}

	@Test
	void filtersByScoreThreshold() {
		RerankingDocumentPostProcessor processor = RerankingDocumentPostProcessor.builder()
			.rerankModel(this.stubModel)
			.scoreThreshold(0.4)
			.build();

		List<Document> result = processor.process(new Query("q"), List.of(doc("low"), doc("high"), doc("mid")));

		assertThat(result).extracting(Document::getText).containsExactly("high", "mid");
	}

	@Test
	void thresholdAndTopNCombine() {
		RerankingDocumentPostProcessor processor = RerankingDocumentPostProcessor.builder()
			.rerankModel(this.stubModel)
			.scoreThreshold(0.4)
			.topN(1)
			.build();

		List<Document> result = processor.process(new Query("q"), List.of(doc("low"), doc("high"), doc("mid")));

		assertThat(result).extracting(Document::getText).containsExactly("high");
	}

	@Test
	void emptyInputPassesThrough() {
		RerankingDocumentPostProcessor processor = RerankingDocumentPostProcessor.builder()
			.rerankModel(this.stubModel)
			.build();

		List<Document> result = processor.process(new Query("q"), List.of());

		assertThat(result).isEmpty();
	}

	@Test
	void builderRequiresRerankModel() {
		assertThatThrownBy(() -> RerankingDocumentPostProcessor.builder().build())
			.isInstanceOf(IllegalArgumentException.class);
	}

	private static Document doc(String text) {
		return new Document(text);
	}

	/**
	 * A stub rerank model that scores documents based on a pre-configured map. Returns
	 * results unsorted to exercise the processor's sorting logic.
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

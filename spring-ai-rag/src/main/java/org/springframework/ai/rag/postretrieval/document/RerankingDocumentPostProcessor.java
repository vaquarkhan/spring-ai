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

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rerank.DefaultRerankOptions;
import org.springframework.ai.rerank.RerankModel;
import org.springframework.ai.rerank.RerankOptions;
import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;
import org.springframework.ai.rerank.RerankResult;
import org.springframework.util.Assert;

/**
 * A {@link DocumentPostProcessor} that reranks retrieved documents using a
 * {@link RerankModel}. Documents are scored for relevance to the query, sorted by
 * descending score, optionally filtered by a minimum score threshold, and optionally
 * truncated to a top-N limit.
 *
 * <p>
 * Usage with the {@code RetrievalAugmentationAdvisor}:
 *
 * <pre>{@code
 * RetrievalAugmentationAdvisor.builder()
 *     .documentRetriever(retriever)
 *     .documentPostProcessors(
 *         RerankingDocumentPostProcessor.builder()
 *             .rerankModel(rerankModel)
 *             .topN(5)
 *             .build())
 *     .build();
 * }</pre>
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public final class RerankingDocumentPostProcessor implements DocumentPostProcessor {

	private final RerankModel rerankModel;

	private final @Nullable Integer topN;

	private final @Nullable Double scoreThreshold;

	private final @Nullable String model;

	private RerankingDocumentPostProcessor(RerankModel rerankModel, @Nullable Integer topN,
			@Nullable Double scoreThreshold, @Nullable String model) {
		Assert.notNull(rerankModel, "rerankModel must not be null");
		this.rerankModel = rerankModel;
		this.topN = topN;
		this.scoreThreshold = scoreThreshold;
		this.model = model;
	}

	@Override
	public List<Document> process(Query query, List<Document> documents) {
		if (documents.isEmpty()) {
			return List.of();
		}

		RerankOptions options = buildOptions();
		RerankRequest request = new RerankRequest(query.text(), documents, options);
		RerankResponse response = this.rerankModel.call(request);

		List<RerankResult> results = response.getResults()
			.stream()
			.sorted(Comparator.comparingDouble(RerankResult::getScore).reversed())
			.collect(Collectors.toList());

		if (this.scoreThreshold != null) {
			results = results.stream().filter(r -> r.getScore() >= this.scoreThreshold).collect(Collectors.toList());
		}

		if (this.topN != null && results.size() > this.topN) {
			results = results.subList(0, this.topN);
		}

		return results.stream().map(RerankResult::getOutput).collect(Collectors.toList());
	}

	private @Nullable RerankOptions buildOptions() {
		if (this.model == null && this.topN == null && this.scoreThreshold == null) {
			return null;
		}
		DefaultRerankOptions.Builder builder = DefaultRerankOptions.builder();
		if (this.model != null) {
			builder.model(this.model);
		}
		if (this.topN != null) {
			builder.topN(this.topN);
		}
		if (this.scoreThreshold != null) {
			builder.scoreThreshold(this.scoreThreshold);
		}
		return builder.build();
	}

	/**
	 * Creates a new builder for {@link RerankingDocumentPostProcessor}.
	 * @return a new builder
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Builder for {@link RerankingDocumentPostProcessor}.
	 */
	public static final class Builder {

		private @Nullable RerankModel rerankModel;

		private @Nullable Integer topN;

		private @Nullable Double scoreThreshold;

		private @Nullable String model;

		private Builder() {
		}

		/**
		 * Sets the rerank model to use for scoring documents.
		 * @param rerankModel the rerank model
		 * @return this builder
		 */
		public Builder rerankModel(RerankModel rerankModel) {
			this.rerankModel = rerankModel;
			return this;
		}

		/**
		 * Sets the maximum number of top-ranked documents to return.
		 * @param topN the top-N limit
		 * @return this builder
		 */
		public Builder topN(int topN) {
			this.topN = topN;
			return this;
		}

		/**
		 * Sets the minimum score threshold. Documents scoring below this value are
		 * excluded.
		 * @param scoreThreshold the score threshold
		 * @return this builder
		 */
		public Builder scoreThreshold(double scoreThreshold) {
			this.scoreThreshold = scoreThreshold;
			return this;
		}

		/**
		 * Sets the model identifier to pass to the rerank model.
		 * @param model the model identifier
		 * @return this builder
		 */
		public Builder model(String model) {
			this.model = model;
			return this;
		}

		/**
		 * Builds the {@link RerankingDocumentPostProcessor}.
		 * @return the post-processor instance
		 * @throws IllegalArgumentException if {@code rerankModel} is not set
		 */
		public RerankingDocumentPostProcessor build() {
			Assert.notNull(this.rerankModel, "rerankModel must not be null");
			return new RerankingDocumentPostProcessor(this.rerankModel, this.topN, this.scoreThreshold, this.model);
		}

	}

}

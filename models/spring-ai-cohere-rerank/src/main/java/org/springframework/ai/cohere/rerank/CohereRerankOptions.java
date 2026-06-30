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

import org.jspecify.annotations.Nullable;

import org.springframework.ai.rerank.RerankOptions;

/**
 * Cohere-specific reranking options. Extends the portable {@link RerankOptions} with
 * Cohere-specific parameters.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public final class CohereRerankOptions implements RerankOptions {

	public static final String DEFAULT_MODEL = "rerank-v3.5";

	private final @Nullable String model;

	private final @Nullable Integer topN;

	private final @Nullable Double scoreThreshold;

	private final @Nullable Integer maxChunksPerDoc;

	private CohereRerankOptions(@Nullable String model, @Nullable Integer topN, @Nullable Double scoreThreshold,
			@Nullable Integer maxChunksPerDoc) {
		this.model = model;
		this.topN = topN;
		this.scoreThreshold = scoreThreshold;
		this.maxChunksPerDoc = maxChunksPerDoc;
	}

	@Override
	public @Nullable String getModel() {
		return this.model;
	}

	@Override
	public @Nullable Integer getTopN() {
		return this.topN;
	}

	@Override
	public @Nullable Double getScoreThreshold() {
		return this.scoreThreshold;
	}

	/**
	 * Returns the maximum number of chunks per document to consider.
	 * @return the max chunks per doc, or {@code null} for the Cohere default
	 */
	public @Nullable Integer getMaxChunksPerDoc() {
		return this.maxChunksPerDoc;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static final class Builder {

		private @Nullable String model;

		private @Nullable Integer topN;

		private @Nullable Double scoreThreshold;

		private @Nullable Integer maxChunksPerDoc;

		private Builder() {
		}

		public Builder model(String model) {
			this.model = model;
			return this;
		}

		public Builder topN(int topN) {
			this.topN = topN;
			return this;
		}

		public Builder scoreThreshold(double scoreThreshold) {
			this.scoreThreshold = scoreThreshold;
			return this;
		}

		public Builder maxChunksPerDoc(int maxChunksPerDoc) {
			this.maxChunksPerDoc = maxChunksPerDoc;
			return this;
		}

		public CohereRerankOptions build() {
			return new CohereRerankOptions(this.model, this.topN, this.scoreThreshold, this.maxChunksPerDoc);
		}

	}

}

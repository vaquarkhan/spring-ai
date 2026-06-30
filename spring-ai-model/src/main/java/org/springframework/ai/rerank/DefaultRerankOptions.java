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

import org.jspecify.annotations.Nullable;

/**
 * Default implementation of {@link RerankOptions} with a fluent builder.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public final class DefaultRerankOptions implements RerankOptions {

	private final @Nullable String model;

	private final @Nullable Integer topN;

	private final @Nullable Double scoreThreshold;

	private DefaultRerankOptions(@Nullable String model, @Nullable Integer topN, @Nullable Double scoreThreshold) {
		this.model = model;
		this.topN = topN;
		this.scoreThreshold = scoreThreshold;
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
	 * Creates a new builder for {@link DefaultRerankOptions}.
	 * @return a new builder instance
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Builder for {@link DefaultRerankOptions}.
	 */
	public static final class Builder {

		private @Nullable String model;

		private @Nullable Integer topN;

		private @Nullable Double scoreThreshold;

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

		public DefaultRerankOptions build() {
			return new DefaultRerankOptions(this.model, this.topN, this.scoreThreshold);
		}

	}

}

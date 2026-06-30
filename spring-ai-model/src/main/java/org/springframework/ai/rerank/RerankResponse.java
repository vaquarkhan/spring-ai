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

import org.jspecify.annotations.Nullable;

import org.springframework.ai.model.ModelResponse;
import org.springframework.ai.model.MutableResponseMetadata;
import org.springframework.ai.model.ResponseMetadata;

/**
 * The response from a reranking model, containing a list of {@link RerankResult} entries.
 * Results are NOT guaranteed to be pre-sorted; consumers should sort by score descending
 * if ordering matters.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public class RerankResponse implements ModelResponse<RerankResult> {

	private final List<RerankResult> results;

	private final ResponseMetadata metadata;

	/**
	 * Creates a new {@link RerankResponse} with default metadata.
	 * @param results the reranked results
	 */
	public RerankResponse(List<RerankResult> results) {
		this(results, new MutableResponseMetadata());
	}

	/**
	 * Creates a new {@link RerankResponse}.
	 * @param results the reranked results
	 * @param metadata the response metadata
	 */
	public RerankResponse(List<RerankResult> results, ResponseMetadata metadata) {
		this.results = results;
		this.metadata = metadata;
	}

	@Override
	public @Nullable RerankResult getResult() {
		return this.results.isEmpty() ? null : this.results.get(0);
	}

	@Override
	public List<RerankResult> getResults() {
		return this.results;
	}

	@Override
	public ResponseMetadata getMetadata() {
		return this.metadata;
	}

}

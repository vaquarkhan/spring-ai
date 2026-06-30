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

import org.springframework.ai.model.ModelOptions;

/**
 * Options for configuring a reranking model call.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public interface RerankOptions extends ModelOptions {

	/**
	 * Returns the model identifier to use for reranking.
	 * @return the model identifier, or {@code null} to use the provider default
	 */
	@Nullable String getModel();

	/**
	 * Returns the maximum number of top-ranked documents to return.
	 * @return the top-N limit, or {@code null} to return all documents
	 */
	@Nullable Integer getTopN();

	/**
	 * Returns the minimum relevance score threshold. Documents scoring below this
	 * threshold are excluded from the results.
	 * @return the score threshold, or {@code null} for no threshold filtering
	 */
	@Nullable Double getScoreThreshold();

}

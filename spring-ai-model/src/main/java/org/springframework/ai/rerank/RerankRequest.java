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

import org.springframework.ai.document.Document;
import org.springframework.ai.model.ModelOptions;
import org.springframework.ai.model.ModelRequest;
import org.springframework.util.Assert;

/**
 * A request to rerank a list of documents against a query. The documents are provided as
 * the {@link #getInstructions() instructions} and the query text is accessible via
 * {@link #getQuery()}.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public class RerankRequest implements ModelRequest<List<Document>> {

	private final String query;

	private final List<Document> documents;

	private final @Nullable RerankOptions options;

	/**
	 * Creates a new {@link RerankRequest}.
	 * @param query the query text to rank documents against
	 * @param documents the candidate documents to rerank
	 * @param options the reranking options, or {@code null} for defaults
	 */
	public RerankRequest(String query, List<Document> documents, @Nullable RerankOptions options) {
		Assert.hasText(query, "query must not be null or blank");
		Assert.notNull(documents, "documents must not be null");
		this.query = query;
		this.documents = documents;
		this.options = options;
	}

	/**
	 * Returns the query text.
	 * @return the query
	 */
	public String getQuery() {
		return this.query;
	}

	/**
	 * Returns the candidate documents to rerank. This is the same as
	 * {@link #getInstructions()}.
	 * @return the documents
	 */
	public List<Document> getDocuments() {
		return this.documents;
	}

	@Override
	public List<Document> getInstructions() {
		return this.documents;
	}

	@Override
	public @Nullable ModelOptions getOptions() {
		return this.options;
	}

	/**
	 * Returns the rerank-specific options.
	 * @return the rerank options, or {@code null} if not set
	 */
	public @Nullable RerankOptions getRerankOptions() {
		return this.options;
	}

}

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

import org.springframework.ai.document.Document;
import org.springframework.ai.model.ModelResult;
import org.springframework.ai.model.ResultMetadata;
import org.springframework.util.Assert;

/**
 * Represents a single reranked document with its relevance score and original position.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public class RerankResult implements ModelResult<Document> {

	private final Document document;

	private final double score;

	private final int index;

	private final ResultMetadata metadata;

	/**
	 * Creates a new {@link RerankResult}.
	 * @param document the reranked document
	 * @param score the relevance score (higher is more relevant)
	 * @param index the original position of the document in the request
	 */
	public RerankResult(Document document, double score, int index) {
		this(document, score, index, RerankResultMetadata.EMPTY);
	}

	/**
	 * Creates a new {@link RerankResult} with metadata.
	 * @param document the reranked document
	 * @param score the relevance score (higher is more relevant)
	 * @param index the original position of the document in the request
	 * @param metadata the result metadata
	 */
	public RerankResult(Document document, double score, int index, ResultMetadata metadata) {
		Assert.notNull(document, "document must not be null");
		Assert.notNull(metadata, "metadata must not be null");
		this.document = document;
		this.score = score;
		this.index = index;
		this.metadata = metadata;
	}

	@Override
	public Document getOutput() {
		return this.document;
	}

	/**
	 * Returns the relevance score for this document. Higher scores indicate greater
	 * relevance to the query.
	 * @return the relevance score
	 */
	public double getScore() {
		return this.score;
	}

	/**
	 * Returns the original position of this document in the input list.
	 * @return the zero-based index
	 */
	public int getIndex() {
		return this.index;
	}

	@Override
	public ResultMetadata getMetadata() {
		return this.metadata;
	}

}

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

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.ai.document.Document;
import org.springframework.ai.model.Model;

/**
 * A model that reranks documents by relevance to a given query. Implementations score
 * each candidate document and return {@link RerankResult} entries with relevance scores.
 *
 * <p>
 * This interface follows the standard {@link Model} contract and can be plugged into the
 * RAG pipeline via
 * {@link org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor}.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public interface RerankModel extends Model<RerankRequest, RerankResponse> {

	@Override
	RerankResponse call(RerankRequest request);

	/**
	 * Convenience method that reranks the given documents against the query and returns
	 * them sorted by descending relevance score.
	 * @param query the query text
	 * @param documents the candidate documents
	 * @return documents sorted by descending relevance score, or an empty list if input
	 * is empty
	 */
	default List<Document> rerank(String query, List<Document> documents) {
		if (documents.isEmpty()) {
			return List.of();
		}
		RerankResponse response = call(new RerankRequest(query, documents, null));
		return response.getResults()
			.stream()
			.sorted(Comparator.comparingDouble(RerankResult::getScore).reversed())
			.map(RerankResult::getOutput)
			.collect(Collectors.toList());
	}

}

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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.jspecify.annotations.Nullable;

import org.springframework.ai.document.Document;
import org.springframework.ai.rerank.RerankModel;
import org.springframework.ai.rerank.RerankOptions;
import org.springframework.ai.rerank.RerankRequest;
import org.springframework.ai.rerank.RerankResponse;
import org.springframework.ai.rerank.RerankResult;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.util.Assert;
import org.springframework.web.client.RestClient;

/**
 * Cohere Rerank API backend for {@link RerankModel}. Calls the Cohere
 * <a href="https://docs.cohere.com/reference/rerank">rerank endpoint</a> to score
 * documents by relevance to a query.
 *
 * <p>
 * Requires a valid Cohere API key. Set via constructor or environment variable
 * {@code COHERE_API_KEY}.
 *
 * @author Vaquar Khan
 * @since 2.0.0
 */
public final class CohereRerankModel implements RerankModel {

	private static final String DEFAULT_BASE_URL = "https://api.cohere.com/v2";

	private final RestClient restClient;

	private final CohereRerankOptions defaultOptions;

	/**
	 * Creates a new {@link CohereRerankModel} with the given API key and default options.
	 * @param apiKey the Cohere API key
	 */
	public CohereRerankModel(String apiKey) {
		this(apiKey, CohereRerankOptions.builder().model(CohereRerankOptions.DEFAULT_MODEL).build());
	}

	/**
	 * Creates a new {@link CohereRerankModel} with the given API key and options.
	 * @param apiKey the Cohere API key
	 * @param defaultOptions the default reranking options
	 */
	public CohereRerankModel(String apiKey, CohereRerankOptions defaultOptions) {
		this(apiKey, DEFAULT_BASE_URL, defaultOptions);
	}

	/**
	 * Creates a new {@link CohereRerankModel} with a custom base URL.
	 * @param apiKey the Cohere API key
	 * @param baseUrl the Cohere API base URL
	 * @param defaultOptions the default reranking options
	 */
	public CohereRerankModel(String apiKey, String baseUrl, CohereRerankOptions defaultOptions) {
		Assert.hasText(apiKey, "apiKey must not be null or blank");
		Assert.hasText(baseUrl, "baseUrl must not be null or blank");
		Assert.notNull(defaultOptions, "defaultOptions must not be null");
		this.defaultOptions = defaultOptions;
		this.restClient = RestClient.builder()
			.baseUrl(baseUrl)
			.defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
			.defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
			.build();
	}

	/**
	 * Package-private constructor for testing with a pre-built RestClient.
	 */
	CohereRerankModel(RestClient restClient, CohereRerankOptions defaultOptions) {
		this.restClient = restClient;
		this.defaultOptions = defaultOptions;
	}

	@Override
	public RerankResponse call(RerankRequest request) {
		Assert.notNull(request, "request must not be null");
		List<Document> documents = request.getDocuments();
		if (documents.isEmpty()) {
			return new RerankResponse(List.of());
		}

		CohereRerankOptions effectiveOptions = mergeOptions(request.getRerankOptions());
		CohereApiRequest apiRequest = buildApiRequest(request.getQuery(), documents, effectiveOptions);

		CohereApiResponse apiResponse = this.restClient.post()
			.uri("/rerank")
			.body(apiRequest)
			.retrieve()
			.body(CohereApiResponse.class);

		return mapResponse(apiResponse, documents);
	}

	private CohereRerankOptions mergeOptions(@Nullable RerankOptions requestOptions) {
		if (requestOptions == null) {
			return this.defaultOptions;
		}
		// Request options override defaults
		CohereRerankOptions.Builder builder = CohereRerankOptions.builder();
		builder.model(
				requestOptions.getModel() != null ? requestOptions.getModel() : (this.defaultOptions.getModel() != null
						? this.defaultOptions.getModel() : CohereRerankOptions.DEFAULT_MODEL));
		if (requestOptions.getTopN() != null) {
			builder.topN(requestOptions.getTopN());
		}
		else if (this.defaultOptions.getTopN() != null) {
			builder.topN(this.defaultOptions.getTopN());
		}
		if (requestOptions.getScoreThreshold() != null) {
			builder.scoreThreshold(requestOptions.getScoreThreshold());
		}
		else if (this.defaultOptions.getScoreThreshold() != null) {
			builder.scoreThreshold(this.defaultOptions.getScoreThreshold());
		}
		if (this.defaultOptions.getMaxChunksPerDoc() != null) {
			builder.maxChunksPerDoc(this.defaultOptions.getMaxChunksPerDoc());
		}
		return builder.build();
	}

	private CohereApiRequest buildApiRequest(String query, List<Document> documents, CohereRerankOptions options) {
		List<String> texts = documents.stream().map(doc -> doc.getText() != null ? doc.getText() : "").toList();

		CohereApiRequest apiRequest = new CohereApiRequest();
		apiRequest.setQuery(query);
		apiRequest.setDocuments(texts);
		apiRequest.setModel(options.getModel() != null ? options.getModel() : CohereRerankOptions.DEFAULT_MODEL);
		if (options.getTopN() != null) {
			apiRequest.setTopN(options.getTopN());
		}
		if (options.getMaxChunksPerDoc() != null) {
			apiRequest.setMaxChunksPerDoc(options.getMaxChunksPerDoc());
		}
		return apiRequest;
	}

	private RerankResponse mapResponse(@Nullable CohereApiResponse apiResponse, List<Document> documents) {
		if (apiResponse == null || apiResponse.getResults() == null) {
			return new RerankResponse(List.of());
		}
		List<RerankResult> results = new ArrayList<>();
		for (CohereApiResponse.Result r : apiResponse.getResults()) {
			int index = r.getIndex();
			Document doc = (index >= 0 && index < documents.size()) ? documents.get(index) : documents.get(0);
			results.add(new RerankResult(doc, r.getRelevanceScore(), index));
		}
		return new RerankResponse(results);
	}

	/**
	 * Internal request structure matching the Cohere Rerank API.
	 */
	static class CohereApiRequest {

		private String query = "";

		private List<String> documents = List.of();

		private String model = CohereRerankOptions.DEFAULT_MODEL;

		private @Nullable Integer topN;

		private @Nullable Integer maxChunksPerDoc;

		public String getQuery() {
			return this.query;
		}

		public void setQuery(String query) {
			this.query = query;
		}

		public List<String> getDocuments() {
			return this.documents;
		}

		public void setDocuments(List<String> documents) {
			this.documents = documents;
		}

		public String getModel() {
			return this.model;
		}

		public void setModel(String model) {
			this.model = model;
		}

		public @Nullable Integer getTopN() {
			return this.topN;
		}

		public void setTopN(@Nullable Integer topN) {
			this.topN = topN;
		}

		public @Nullable Integer getMaxChunksPerDoc() {
			return this.maxChunksPerDoc;
		}

		public void setMaxChunksPerDoc(@Nullable Integer maxChunksPerDoc) {
			this.maxChunksPerDoc = maxChunksPerDoc;
		}

	}

	/**
	 * Internal response structure matching the Cohere Rerank API.
	 */
	static class CohereApiResponse {

		private @Nullable List<Result> results;

		private @Nullable Map<String, Object> meta;

		public @Nullable List<Result> getResults() {
			return this.results;
		}

		public void setResults(@Nullable List<Result> results) {
			this.results = results;
		}

		public @Nullable Map<String, Object> getMeta() {
			return this.meta;
		}

		public void setMeta(@Nullable Map<String, Object> meta) {
			this.meta = meta;
		}

		static class Result {

			private int index;

			private double relevanceScore;

			public int getIndex() {
				return this.index;
			}

			public void setIndex(int index) {
				this.index = index;
			}

			public double getRelevanceScore() {
				return this.relevanceScore;
			}

			public void setRelevanceScore(double relevanceScore) {
				this.relevanceScore = relevanceScore;
			}

		}

	}

}

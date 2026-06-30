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

/**
 * Provides a portable abstraction for document reranking models. The
 * {@link org.springframework.ai.rerank.RerankModel} interface follows the standard
 * {@link org.springframework.ai.model.Model} contract and can be used with the RAG
 * pipeline's {@code DocumentPostProcessor} extension point.
 */
@NullMarked
package org.springframework.ai.rerank;

import org.jspecify.annotations.NullMarked;

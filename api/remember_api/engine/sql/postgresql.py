"""PostgreSQL SQL dialect implementation.

Provides PostgreSQL-specific SQL fragments for parameter binding, JSON operators,
vector distance (pgvector), native PostgreSQL full-text search,
and other non-portable patterns.
"""

from dataclasses import dataclass

from .base import bm25_score_gate


@dataclass(frozen=True)
class PageBm25Arm:
    """Native Page text-search ranking and match clauses."""

    order_by: str
    match_filter: str


def page_bm25_arm(*, table_alias: str, text_param: str) -> PageBm25Arm:
    """Build the native full-text ranking and match clauses for Page search."""
    # The Page tsvector is generated with the English dictionary. Disjunctive
    # query terms match recall's candidate generation and avoid empty results
    # for natural-language questions.
    score = f"ts_rank_cd({table_alias}.search_vector, to_tsquery('english', {text_param}))"
    return PageBm25Arm(
        order_by=f"{score} DESC",
        match_filter=f"AND {table_alias}.search_vector @@ to_tsquery('english', {text_param})",
    )


class PostgreSQLDialect:
    """SQL dialect for PostgreSQL (asyncpg)."""

    # -- Parameter binding -----------------------------------------------

    def param(self, n: int) -> str:
        return f"${n}"

    # -- Type casting ----------------------------------------------------

    def cast(self, param: str, type_name: str) -> str:
        return f"{param}::{type_name}"

    # -- Vector operations -----------------------------------------------

    def vector_distance(self, col: str, param: str) -> str:
        return f"{col} <=> {param}::vector"

    def vector_similarity(self, col: str, param: str) -> str:
        return f"1 - ({col} <=> {param}::vector)"

    # -- JSON operations -------------------------------------------------

    def json_extract_text(self, col: str, key: str) -> str:
        return f"{col} ->> '{key}'"

    def json_contains(self, col: str, param: str) -> str:
        return f"{col} @> {param}::jsonb"

    def json_merge(self, col: str, param: str) -> str:
        return f"{col} || {param}::jsonb"

    # -- Text search -----------------------------------------------------

    def text_search_score(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        if index_name:
            # VectorChord BM25
            return f"-({col} <@> to_bm25query({query_param}, '{index_name}'))"
        # Fallback to tsvector
        return f"ts_rank_cd({col}, to_tsquery({query_param}))"

    def text_search_order(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        if index_name:
            # VectorChord BM25 — lower distance = better, so ASC
            return f"{col} <@> to_bm25query({query_param}, '{index_name}') ASC"
        return f"ts_rank_cd({col}, to_tsquery({query_param})) DESC"

    # -- Fuzzy string matching -------------------------------------------

    def similarity(self, col: str, param: str) -> str:
        return f"similarity({col}, {param})"

    # -- Upsert ----------------------------------------------------------

    def upsert(
        self,
        table: str,
        columns: list[str],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> str:
        col_list = ", ".join(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        conflict = ", ".join(conflict_columns)

        if not update_columns:
            return f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO NOTHING"

        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
        return (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        )

    # -- Bulk operations -------------------------------------------------

    def bulk_unnest(self, param_types: list[tuple[str, str]]) -> str:
        args = ", ".join(f"{p}::{t}" for p, t in param_types)
        return f"unnest({args})"

    # -- Pagination ------------------------------------------------------

    def limit_offset(self, limit_param: str, offset_param: str) -> str:
        return f"LIMIT {limit_param} OFFSET {offset_param}"

    # -- RETURNING clause ------------------------------------------------

    def returning(self, columns: list[str]) -> str:
        return f"RETURNING {', '.join(columns)}"

    # -- Pattern matching ------------------------------------------------

    def ilike(self, col: str, param: str) -> str:
        return f"{col} ILIKE {param}"

    # -- Array operations ------------------------------------------------

    def array_any(self, param: str) -> str:
        return f"= ANY({param})"

    def array_all(self, param: str) -> str:
        return f"!= ALL({param})"

    def array_contains(self, col: str, param: str) -> str:
        return f"{col} @> {param}::varchar[]"

    # -- Locking ---------------------------------------------------------

    def for_update_skip_locked(self) -> str:
        return "FOR UPDATE SKIP LOCKED"

    # -- UUID generation -------------------------------------------------

    def generate_uuid(self) -> str:
        return "gen_random_uuid()"

    # -- Misc ------------------------------------------------------------

    def greatest(self, *args: str) -> str:
        return f"GREATEST({', '.join(args)})"

    def current_timestamp(self) -> str:
        return "now()"

    def array_agg(self, expr: str) -> str:
        return f"array_agg({expr})"

    # -- Retrieval query arms ----------------------------------------------

    def build_semantic_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        embedding_param: str,
        bank_id_param: str,
        fetch_limit: int,
        min_similarity: float,
        tags_clause: str = "",
        groups_clause: str = "",
        extra_where: str = "",
    ) -> str:
        return (
            f"(SELECT {cols},"
            f"        1 - (embedding <=> {embedding_param}::vector) AS similarity,"
            f"        NULL::float AS bm25_score,"
            f"        'semantic' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            f"   AND embedding IS NOT NULL"
            f"   AND (1 - (embedding <=> {embedding_param}::vector)) >= {min_similarity}"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY embedding <=> {embedding_param}::vector"
            f" LIMIT {fetch_limit})"
        )

    def build_bm25_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        bank_id_param: str,
        limit_param: str,
        text_param: str,
        tags_clause: str = "",
        groups_clause: str = "",
        arm_index: int = 0,
        text_search_extension: str = "native",
        bm25_language: str = "english",
        bm25_min_score: float = 0.0,
        extra_where: str = "",
    ) -> str:
        if text_search_extension != "native":
            raise ValueError(f"Unsupported text search extension: {text_search_extension}")
        # `bm25_language` is validated as a PostgreSQL identifier in config.
        bm25_score_expr = f"ts_rank_cd(search_vector, to_tsquery('{bm25_language}', {text_param}))"
        bm25_order_by = f"{bm25_score_expr} DESC, id"
        bm25_where_filter = f"AND search_vector @@ to_tsquery('{bm25_language}', {text_param})"

        arm = (
            f"(SELECT {cols},"
            f"        NULL::float AS similarity,"
            f"        {bm25_score_expr} AS bm25_score,"
            f"        'bm25' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            f"   {bm25_where_filter}"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY {bm25_order_by}"
            f" LIMIT {limit_param})"
        )

        if bm25_min_score <= 0:
            return arm
        # Filter the ordered LIMIT slice to avoid computing the score twice.
        return f"(SELECT * FROM {arm} AS bm25_arm_{arm_index} WHERE bm25_score {bm25_score_gate(bm25_min_score)})"

    def prepare_bm25_text(
        self,
        tokens: list[str],
        query_text: str,
        *,
        text_search_extension: str = "native",
        max_query_terms: int | None = None,
    ) -> str:
        if text_search_extension != "native":
            raise ValueError(f"Unsupported text search extension: {text_search_extension}")
        if max_query_terms is not None and max_query_terms > 0:
            tokens = tokens[:max_query_terms]
        # native tsvector: join tokens with OR operator
        return " | ".join(tokens)

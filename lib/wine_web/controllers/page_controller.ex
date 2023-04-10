defmodule WineWeb.PageController do
  use WineWeb, :controller

  alias Tokenizers.{Tokenizer, Encoding}

  @elasticsearch_endpoint "https://localhost:9200/wine/_knn_search"
  @cacertfile_path "http_ca.crt"
  @top_k 5
  @num_candidates 100

  def home(conn, %{"query" => query}) do
    {predict_fn, params} = get_model()

    inputs = get_inputs_from_query(query)

    embedded_vector =
      predict_fn.(params, inputs)
      |> Nx.to_flat_list()

    case get_closest_results(embedded_vector) do
      {:ok, documents} ->
        render(conn, :home, wine_documents: documents, query: %{"query" => query})

      _error ->
        conn
        |> put_flash(:error, "Something went wrong!")
        |> render(:home, wine_documents: [], query: %{})
    end
  end

  def home(conn, _params) do
    # The home page is often custom made,
    # so skip the default app layout.
    render(conn, :home, wine_documents: [], query: %{})
  end

  defp get_model do
    :persistent_term.get({Wine.Model, :model})
  end

  defp get_tokenizer do
    :persistent_term.get({Wine.Model, :tokenizer})
  end

  defp get_inputs_from_query(query) do
    tokenzier = get_tokenizer()

    {:ok, encoded_seq} = Tokenizer.encode(tokenzier, query)

    encoded_seq =
      encoded_seq
      |> Encoding.pad(Wine.Model.max_sequence_length())
      |> Encoding.truncate(Wine.Model.max_sequence_length())

    input_ids = encoded_seq |> Encoding.get_ids() |> Nx.tensor()
    token_type_ids = encoded_seq |> Encoding.get_type_ids() |> Nx.tensor()
    attention_mask = encoded_seq |> Encoding.get_attention_mask() |> Nx.tensor()

    %{
      "input_ids" => Nx.new_axis(input_ids, 0),
      "token_type_ids" => Nx.new_axis(token_type_ids, 0),
      "attention_mask" => Nx.new_axis(attention_mask, 0)
    }
  end

  defp get_closest_results(embedded_vector) do
    options = [ssl: [cacertfile: @cacertfile_path], recv_timeout: 60_000]

    password = System.get_env("ELASTICSEARCH_PASSWORD")
    credentials = "elastic:#{password}"

    headers = [
      Authorization: "Basic #{Base.encode64(credentials)}",
      "Content-Type": "application/json"
    ]

    query = format_query(embedded_vector)

    with {:ok, data} <- Jason.encode(query),
         {:ok, response} <- HTTPoison.post(@elasticsearch_endpoint, data, headers, options),
         {:ok, results} <- Jason.decode(response.body) do
      parse_response(results)
    else
      _error ->
        :error
    end
  end

  defp format_query(vector) do
    %{
      "knn" => %{
        "field" => "document-vector",
        "query_vector" => vector,
        "k" => @top_k,
        "num_candidates" => @num_candidates
      },
      "_source" => ["url"]
    }
  end

  defp parse_response(response) do
    hits = get_in(response, ["hits", "hits"])

    case hits do
      nil ->
        :error

      hits ->
        results =
          Enum.map(hits, fn
            %{"_source" => result} ->
              url = result["url"]
              get_wine_preview(url)
          end)

        {:ok, results}
    end
  end

  defp get_wine_preview(url) do
    url = String.replace(url, "//product", "/product")

    with {:ok, %{body: body}} <- HTTPoison.get(url),
         {:ok, page} <- Floki.parse_document(body) do
      title = page |> Floki.find(".pipName") |> Floki.text()
      %{url: url, title: title}
    else
      _error ->
        %{url: url, title: "Generic wine"}
    end
  end
end

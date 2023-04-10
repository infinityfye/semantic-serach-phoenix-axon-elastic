Mix.install([
  {:httpoison, "~> 2.1"},
  {:jason, "~> 1.4"},
  {:axon, "~> 0.5.1"},
  {:axon_onnx, "~> 0.4.0"},
  {:exla, "~> 0.5.2"},
  {:nx, "~> 0.5.2"},
  {:tokenizers, "~> 0.3.1"},
  {:rustler, "~> 0.27.0"}
])

defmodule EmbedWineDocuments do
  alias Tokenizers.{Tokenizer, Encoding}

  def format_document(document) do
    "Name: #{document["name"]}\n" <>
      "Varietal: #{document["varietal"]}\n" <>
      "Location: #{document["location"]}\n" <>
      "Alcohol Volume: #{document["alcohol_volume"]}\n" <>
      "Alcohol Percent: #{document["alcohol_percent"]}\n" <>
      "Price: #{document["price"]}\n" <>
      "Winemaker Notes: #{document["notes"]}\n" <>
      "Reviews:\n#{format_reviews(document["reviews"])}"
  end

  defp format_reviews(reviews) do
    reviews
    |> Enum.map(fn review ->
      "Reviewer: #{review["author"]}\n" <>
        "Review: #{review["review"]}\n" <>
        "Rating: #{review["rating"]}"
    end)
    |> Enum.join("\n")
  end

  def encode_text(tokenizer, text, max_sequence_length) do
    {:ok, encoding} = Tokenizer.encode(tokenizer, text)

    encoded_seq =
      encoding
      |> Enum.map(&Encoding.pad(&1, max_sequence_length))
      |> Enum.map(&Encoding.truncate(&1, max_sequence_length))

    input_ids = encoded_seq |> Enum.map(&Encoding.get_ids/1) |> Nx.tensor()
    token_type_ids = encoded_seq |> Enum.map(&Encoding.get_type_ids/1) |> Nx.tensor()
    attention_mask = encoded_seq |> Enum.map(&Encoding.get_attention_mask/1) |> Nx.tensor()

    %{
      "input_ids" => input_ids,
      "token_type_ids" => token_type_ids,
      "attention_mask" => attention_mask
    }
  end

  def compute_embedding(model, params, inputs) do
    Axon.predict(model, params, inputs, compiler: EXLA)
  end
end

max_sequence_length = 120
batch_size = 128

{bert, bert_params} =
  AxonOnnx.import("priv/models/model.onnx", batch: batch_size, sequence: max_sequence_length)

bert = Axon.nx(bert, fn {_, out} -> out end)

{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

path_to_wines = "priv/wine_documents.jsonl"
endpoint = "https://localhost:9200/wine/_doc/"
password = System.get_env("ELASTICSEARCH_PASSWORD")
credentials = "elastic:#{password}"
headers = [
  Authorization: "Basic #{Base.encode64(credentials)}",
  "Content-Type": "application/json"
]
options = [ssl: [cacertfile: "http_ca.crt"]]

document_stream =
  path_to_wines
  |> File.stream!()
  |> Stream.map(&Jason.decode!/1)
  |> Stream.map(fn document -> {document["url"], EmbedWineDocuments.format_document(document)} end)
  |> Stream.chunk_every(batch_size)
  |> Stream.flat_map(fn batches ->
    {urls, texts} = Enum.unzip(batches)
    inputs = EmbedWineDocuments.encode_text(tokenizer, texts, max_sequence_length)
    embedded = EmbedWineDocuments.compute_embedding(bert, bert_params, inputs)

    embedded
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.to_flat_list(Nx.squeeze(&1)))
    |> Enum.zip_with(urls, fn vec, url -> %{"url" => url, "document-vector" => vec} end)
    |> Enum.map(&Jason.encode!/1)
  end)
  |> Stream.map(fn data ->
    {:ok, _} = HTTPoison.post(endpoint, data, headers, options)
    :ok
  end)

Enum.reduce(document_stream, 0, fn :ok, counter ->
  IO.write("\rDocuments Embedded: #{counter}")
  counter + 1
end)

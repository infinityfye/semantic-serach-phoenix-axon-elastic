defmodule Wine.Model do
  @max_sequence_length 120

  def load() do
    {model, params} =
      AxonOnnx.import("priv/models/model.onnx", batch: 1, sequence: max_sequence_length())

    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

    params = init_fn.(inputs(), params)

    predict_fn =
      EXLA.compile(
        fn params, inputs ->
          {_, pooled} = predict_fn.(params, inputs)
          Nx.squeeze(pooled)
        end,
        [params, inputs()]
      )

    :persistent_term.put({__MODULE__, :model}, {predict_fn, params})
    # Load the tokenizer as well
    :persistent_term.put({__MODULE__, :tokenizer}, tokenizer)

    :ok
  end

  def max_sequence_length(), do: @max_sequence_length

  defp inputs() do
    %{
      "input_ids" => Nx.template({1, 120}, {:s, 64}),
      "token_type_ids" => Nx.template({1, 120}, {:s, 64}),
      "attention_mask" => Nx.template({1, 120}, {:s, 64})
    }
  end
end

# Running Code for Sean Moriarity's Dockyard article "Semantic Search with Phoenix, Axon, and Elastic"

This is running code copied (almost) verbatim from the [article](https://dockyard.com/blog/2022/09/28/semantic-search-with-phoenix-axon-and-elastic)

I have only made small changes to make the code work as in the article:

- `Wine.Model` changed from `Axon.compile/4` to `Axon.build/2`
- [`wine_documents.jsonl`](https://gist.github.com/seanmor5/af60a4a22dfc51250661380975281fa6) As given in the article has a typo in all the `url`s

You will still need to setup elastic search on docker, copy the password and the certificate, run the document seeding script, etc...

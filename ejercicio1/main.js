function calificar() {
    // se le asigna el genero al que corresponde cada grupo
    const generos = ["rock", "trap", "rap", "reggaeton", "bachata"];

    // se le asigna un valor a cada genero
    const valores = tf.tensor2d([
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1],
      [0, 0, 1, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0],
    ]);

    // se captura los datos del usuario y se convierten en un tensor
    const user_votes = tf.tensor2d([
      [
        parseFloat(document.getElementById("Duki").value),
        parseFloat(document.getElementById("NFT").value),
        parseFloat(document.getElementById("LilChip").value),
        parseFloat(document.getElementById("dragons").value),
        parseFloat(document.getElementById("Maluma").value),
        parseFloat(document.getElementById("bionica").value),
        parseFloat(document.getElementById("soda").value),
        parseFloat(document.getElementById("Arcangel").value),
      ],
    ]);

    // se calcula la calificacion
    const calificacion = tf.matMul(user_votes, valores);

    // se obtiene los generos con mayor calificacion
    const top_user_features = tf.topk(calificacion, generos.length);

    const top_genres = top_user_features.indices.arraySync()[0];

    const rankedGenres = top_genres.map((index) => generos[index]);

    // Mostrar el resultado en la página
    const resultadoDiv = document.getElementById("resultado");
    resultadoDiv.innerHTML = `<p>
          <em>
          Según tus respuesta, tus géneros musicales favoritos son: ${rankedGenres.join(", ")}
          </em>
          </p>`;
  }
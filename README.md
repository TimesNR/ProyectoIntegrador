# ProyectoIntegrador

Hola.


Este repositorio será para los archivos del proyecto. Pero antes una breve introducción a github
Para poder usar github adecuadamente es necesario instalar [git](https://github.com/mouredev/hello-git?tab=readme-ov-file) el cual se maneja similar a una interfaz de comandos. En caso de no querer usarla y prefieren algo más intuitivo de usar les recomiendo Github Destktop. 


Para la parte del __bash__ se maneja similar a linux
* ls: Observar los documentos en la carpeta en la cual estas ubicado  
* cd + nombre_carpeta\: Permite moverte entre carpetas
* comando -help: Permite ver las opciones de los comandos
* pwd: Muestra tu ruta de archivos
* mkdir: Crea una carpeta o un archivo
* git log: Muestra el hash del commit
* git add nombre_archivo: Agrega un arc


## 3 Python Style Rules

### General Rules
- No terminar con `;`. No poner dos statements en una misma línea.
- Evitar líneas de más de 80 caracteres.
- Cuando comentes un URL, ponerlo en una línea por separado:
  ```python
  # See details at
  # http://www.example.com/us/developer/documentation/api/content/v2.0/csv_file_name_extension_full_specification
  ```
- No usar paréntesis en el `return` o en la condición del `if`.

---

### 3.4 Indentation
- Indentar con **4 espacios** y NO con tabulador.
- En el caso de diccionario, dejar la llave junto con su valor asociado en una misma línea.
- Para **trailing commas**, se puede separar por elemento, pero el bracket final debe estar en otra línea:
  ```python
  golomb4 = [
      0,
      1,
      4,
      6,
  ]
  ```

---

### 3.5 Blank Lines
- **Dos líneas** entre funciones y clases.
- **Una línea** entre cada definición de métodos.

---

### 3.6 White Spaces
- Sin espacios en blanco entre **paréntesis, comas, dos puntos, etc.**
- **1 espacio en blanco entre operadores lógicos**:
  ```python
  x == 1
  ```
- Al definir argumentos por defecto en una función, usar espacio en blanco entre el operador de asignación `=`.
- **No alinear comentarios.**

---

### 3.8 Comments and Docstrings
- Poner un string describiendo el programa, la clase, etc.
  ```python
  """A one-line summary of the module or program, terminated by a period.
  
  Leave one blank line.  The rest of this docstring should contain an
  overall description of the module or program.  Optionally, it may also
  contain a brief description of exported classes and functions and/or usage
  examples.
  
  Typical usage example:
  
    foo = ClassFoo()
    bar = foo.function_bar()
  """
  ```
- Incluir docstrings en **APIs públicas, funciones largas o con lógica complicada**. Debe tener una sección para los **argumentos** y otra para el **return**.
- Si se hace un **override**, solo es necesario un docstring si la función cambia fundamentalmente.
- Para las **clases**, debe haber un docstring describiéndola y sus atributos públicos. Las funciones de la clase también deben incluir docstrings.
- **Comentar antes de operaciones complicadas**. Si el comentario no describe algo complejo, ponerlo en la misma línea.
- **Usar doble quotes (`"""`) para docstrings.**
- **Formato recomendado para long strings:**
  ```python
  long_string = ("And this too is fine if you cannot accept\n"
                 "extraneous leading spaces.")
  ```
- Usar **TODO comments** en soluciones temporales del código.
- Los `import` se organizan **del más genérico al más específico**.
- No usar getters y setters innecesarios, solo si agregan información relevante.

---

### 3.16 Naming Conventions
- **Los nombres deben ser descriptivos**, excepto en valores iterativos de `for` (ejemplo: `i`, `e`).
- **No usar abreviaciones.**
- Separar por **guion bajo (`snake_case`)**.
- Evitar nombres que incluyan el tipo de la variable innecesariamente.

---

### 3.17 MAIN Execution
Si el archivo está destinado a ser ejecutable, usar la siguiente estructura:
```python
from absl import app
...

def main(argv: Sequence[str]):
    # process non-flag arguments
    ...

if __name__ == '__main__':
    app.run(main)

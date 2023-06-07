## FasterTransformer Python APIs

The easy API for faster transformer.

```python
import fastertransformer as ft
model = "gpt2-xl"
model = ft.init_inference(model, 1, 1)
result = model.pipeline_generate(["Amazon is ", "FasterTransformer is","Deepspeed is"])
print(result)
```

to run with the above code

```
python3 test.py
```

### Build the wheel

```
pip wheel --no-deps .
```

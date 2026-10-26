# Turkish Food Classification with Transfer Learning

An image classifier that recognizes **25 Turkish dishes** (mantı, lahmacun, baklava, karnıyarık, ...) from photos.
Three pretrained CNNs were fine-tuned and compared; the best model reaches **91.4% accuracy** on a held-out test set.
Grad-CAM is used to check *where* the model looks when it makes a decision.

## Dataset

[TurkishFoods-25](https://huggingface.co/datasets/yunusserhat/TurkishFoods-25) (Hugging Face, Apache-2.0)

| Split | Images |
|---|---|
| Train | 9,168 |
| Validation | 1,146 |
| Test | 1,147 |

The classes are moderately imbalanced: the training set has 181 images of *aşure* and 706 of *kebap*.

## Approach

1. **Preprocessing:** All images are resized to 224×224. During training, random horizontal flips, rotations and zooms are applied (data augmentation) to reduce overfitting.
2. **Transfer learning in two stages**, starting from ImageNet weights:
   - **Stage 1:** The backbone is frozen. Only a new classification head (global average pooling → dropout 0.3 → dense softmax) is trained, using Adam with a learning rate of 1e-3.
   - **Stage 2 (fine-tuning):** The last 40 backbone layers are unfrozen, with BatchNorm layers kept frozen. Training continues with Adam at a learning rate of 5e-5.
   - Early stopping on validation accuracy (patience 3) is used in both stages.
3. **Model comparison:** EfficientNetV2-B0, MobileNetV3-Large and ConvNeXt-Tiny.
4. **Ensembles:** A simple average of the three models' probabilities, and a weighted average whose weights were selected on the **validation** set, never on the test set.
5. **Explainability:** Grad-CAM heatmaps from the best model.

## Results

Accuracy on the test set (1,147 images the models never saw during training):

| Model | Test accuracy |
|---|---|
| EfficientNetV2-B0 | 86.1% |
| MobileNetV3-Large | 86.4% |
| **ConvNeXt-Tiny** | **91.4%** |
| Ensemble (simple average) | 90.2% |
| Ensemble (weighted 0.3 / 0.2 / 0.5) | 91.4% |

**Takeaways**

- ConvNeXt-Tiny, the largest and most modern backbone, clearly outperformed the two lighter models.
- The simple-average ensemble beat the two lighter models but not ConvNeXt-Tiny. The two weaker models pulled down its predictions.
- The weighted ensemble only *matched* ConvNeXt-Tiny. Since the ensemble needs three models at inference time, ConvNeXt-Tiny alone is the more practical choice.

## Where does the model look? (Grad-CAM)

Random test examples:

![Grad-CAM examples](images/gradcam_examples.png)

Misclassified test examples:

![Grad-CAM mistakes](images/gradcam_mistakes.png)

The errors were inspected with Grad-CAM:

| True | Predicted | Observation |
|---|---|---|
| Mantı | Pirinç pilavı (61%) | The photo shows uncooked mantı, which looks unlike typical cooked mantı. The small pale pieces resemble rice. |
| Lokum | Baklava (55%) | The model looks at the food itself. Yellow, square, coconut-covered pieces resemble baklava. |
| Et sote | Kebap (45%) | The model focuses on the side salad and fries, not the main dish. It appears to have learned a *context shortcut*, since kebab plates are often served with these sides. |
| Enginar | Börek (65%) | Oven-baked artichokes with a golden cheese crust look like börek. |

- Mistakes mostly happen on dishes shown in unusual ways, and between dishes that look alike.
- The model's confidence on wrong predictions is low (45–65%).

## Limitations and next steps

- Some images contain watermarks (stock-photo and recipe-site logos). These are a possible shortcut for the model.
- The test set is small: one percentage point is about 11 images, so small differences between models may be noise.
- Class imbalance was not explicitly handled. Class weights or more data for small classes could be tried.
- More varied photos of each dish (raw/cooked, different servings) could reduce the context shortcuts seen above.

## How to run

1. Open `turkish_food_classification.ipynb` in [Google Colab](https://colab.research.google.com).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The dataset downloads automatically from Hugging Face. Training all three models takes about 45 minutes on a T4 GPU.

## Tech stack

Python · TensorFlow / Keras · Hugging Face Datasets · NumPy · Matplotlib · scikit-learn · Google Colab

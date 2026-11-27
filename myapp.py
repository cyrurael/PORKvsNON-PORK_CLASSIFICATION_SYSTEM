# ───────────────────────────
# Confusion Matrix — FIXED & SAFE VERSION
# ───────────────────────────
st.subheader("📉 Confusion Matrix")
st.caption("How well the model classifies each category")

# Make sure we have exactly 2 labels
if len(labels) >= 2:
    label_0 = labels[0]
    label_1 = labels[1]
else:
    label_0 = "Non-Pork"
    label_1 = "Pork"
    st.warning("⚠️ labels.txt has fewer than 2 labels — using defaults")

# ┌── REPLACE THESE WITH YOUR ACTUAL TEST RESULTS ──
confusion_matrix = np.array([
    [46, 3],   # Non-Pork: 46 correct, 3 misclassified as Pork
    [2, 49]    # Pork: 49 correct, 2 misclassified as Non-Pork
])

cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

fig_cm = f"""
<style>
.cm-table {{ border-collapse: collapse; width: 100%; font-family: sans-serif; }}
.cm-table td, .cm-table th {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
.cm-table th {{ background-color: #f0f2f6; }}
</style>
<table class="cm-table">
  <tr>
    <th></th>
    <th colspan="2">🔮 Predicted</th>
  </tr>
  <tr>
    <th rowspan="2">✅ Actual</th>
    <th>{label_0}</th>
    <th>{label_1}</th>
  </tr>
  <tr>
    <td style="background-color: rgba(144, 238, 144, {cm_normalized[0,0]:.4f}); font-weight: bold;">
      {confusion_matrix[0,0]}<br><small>{cm_normalized[0,0]*100:.1f}%</small><br><span style="color:green">✓ Correct</span>
    </td>
    <td style="background-color: rgba(255, 160, 122, {cm_normalized[0,1]:.4f});">
      {confusion_matrix[0,1]}<br><small>{cm_normalized[0,1]*100:.1f}%</small><br><span style="color:red">✗ Error</span>
    </td>
  </tr>
  <tr>
    <th>{label_1}</th>
    <td style="background-color: rgba(255, 160, 122, {cm_normalized[1,0]:.4f});">
      {confusion_matrix[1,0]}<br><small>{cm_normalized[1,0]*100:.1f}%</small><br><span style="color:red">✗ Error</span>
    </td>
    <td style="background-color: rgba(144, 238, 144, {cm_normalized[1,1]:.4f}); font-weight: bold;">
      {confusion_matrix[1,1]}<br><small>{cm_normalized[1,1]*100:.1f}%</small><br><span style="color:green">✓ Correct</span>
    </td>
  </tr>
</table>
"""
st.markdown(fig_cm, unsafe_allow_html=True)

st.info("""
**Reading the matrix:**
- 🟢 **Diagonal** = Correct predictions
- 🟠 **Off-diagonal** = Misclassifications/Errors
- Higher diagonal values = Better performance
""")

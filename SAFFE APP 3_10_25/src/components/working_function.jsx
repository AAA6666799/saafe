<section style={{ marginTop: 16 }}>
  <h4 style={h4}>Tensor value Live button</h4>

  {error && <p style={{ color: "red" }}>{error}</p>}

  {/* ----- Status Buttons ----- */}
 {sensorData?.prediction && (
    <div style={{ marginBottom: 16 }}>
      {/* 🔥 FIRE */}
      {sensorData.prediction.fire_probability >= 0.7 && (
        <button
          style={{
            backgroundColor: "#ef4444",
            color: "white",
            padding: "8px 16px",
            marginRight: "8px",
            border: "none",
            borderRadius: "4px",
            fontWeight: "bold",
          }}
        >
          🔥 FIRE
        </button>
      )}

      {/* 🟠 PREDICTED */}
      {sensorData.prediction.fire_probability > 0.2 &&
        sensorData.prediction.fire_probability < 0.7 && (
          <button
            style={{
              backgroundColor: "#eab308",
              color: "black",
              padding: "8px 16px",
              marginRight: "8px",
              border: "none",
              borderRadius: "4px",
              fontWeight: "bold",
            }}
          >
            ⚠️ PREDICTED
          </button>
        )}

      {/* ✅ NON-FIRE */}
      {sensorData.prediction.fire_probability <= 0.2 && (
        <button
          style={{
            backgroundColor: "#22c55e",
            color: "white",
            padding: "8px 16px",
            marginRight: "8px",
            border: "none",
            borderRadius: "4px",
            fontWeight: "bold",
          }}
        >
          ✅ NON-FIRE
        </button>
      )}
    </div>
  )}
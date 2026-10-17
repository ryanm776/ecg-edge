from ecg.export import dequantize_value, quantize_value, write_c_header, write_threshold_header


def test_write_c_header_format(tmp_path):
    data = bytes(range(14))  # 14 bytes to exercise the 12-per-line wrap
    path = tmp_path / "model_data.h"

    write_c_header(data, path, array_name="g_model")
    text = path.read_text()

    assert "#ifndef G_MODEL_DATA_H_" in text
    assert "#define G_MODEL_DATA_H_" in text
    assert text.strip().endswith("#endif  // G_MODEL_DATA_H_")
    assert "const unsigned int g_model_len = 14;" in text
    assert "0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b," in text
    assert "0x0c, 0x0d," in text


def test_write_threshold_header_format(tmp_path):
    path = tmp_path / "threshold.h"

    write_threshold_header(0.3, path)
    text = path.read_text()

    assert "constexpr float kAbnormalThreshold = 0.3000f;" in text
    assert "#ifndef" in text and "#endif" in text


def test_quantize_dequantize_roundtrip_scalar():
    scale, zero_point = 0.01, 5

    quantized = quantize_value(0.5, scale, zero_point)
    assert isinstance(quantized, int)
    assert quantized == 55  # round(0.5/0.01) + 5

    restored = dequantize_value(quantized, scale, zero_point)
    assert abs(restored - 0.5) < 1e-6


def test_quantize_value_clamps_to_int8_range():
    assert quantize_value(1000.0, scale=0.01, zero_point=0) == 127
    assert quantize_value(-1000.0, scale=0.01, zero_point=0) == -128

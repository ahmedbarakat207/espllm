import os

model_path = 'model/model.pt.quantized'
output_path = 'src/model_weights.hpp'

if not os.path.exists('src'):
    os.makedirs('src')

with open(model_path, 'rb') as f:
    data = f.read()

with open(output_path, 'w') as f:
    f.write('#ifndef MODEL_WEIGHTS_HPP\n')
    f.write('#define MODEL_WEIGHTS_HPP\n\n')
    f.write('#include <stdint.h>\n\n')
    
    # Store in PROGMEM (Flash memory)
    f.write('// Auto-generated 4-bit model weights (converted from model.pt.quantized)\n')
    f.write('#ifdef __AVR__\n')
    f.write('#include <avr/pgmspace.h>\n')
    f.write('#else\n')
    f.write('#define PROGMEM\n')
    f.write('#endif\n\n')
    
    f.write(f'const uint8_t model_weights[{len(data)}] PROGMEM = {{\n    ')
    
    for i, byte in enumerate(data):
        f.write(f'0x{byte:02x}')
        if i < len(data) - 1:
            f.write(', ')
        if (i + 1) % 12 == 0:
            f.write('\n    ')
            
    f.write('\n};\n\n')
    f.write(f'const unsigned int model_weights_len = {len(data)};\n\n')
    f.write('#endif // MODEL_WEIGHTS_HPP\n')

print(f"Successfully converted {len(data)} bytes to {output_path}")

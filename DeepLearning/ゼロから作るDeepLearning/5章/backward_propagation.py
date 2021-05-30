# %%
from multi_layer import MulLayer
from add_layer import AddLayer
# %%
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(f'価格：{price}')

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(f'Pdapple, dapple_num, dtax')

#%%
orange = 150
orange_num = 3

#layer
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()

# forward
orange_price = mul_orange_layer(orange, orange_num)
all_price = add_apple_orange_layer(apple_price, orange_price)
price = mul_tax_layer(all_price, tax)

# backward
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)

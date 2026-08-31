import * as React from "react";
import { Button, Flex } from "@mantine/core";
import { ViserInputComponent } from "./common";
import { GuiButtonGroupMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { toMantineColor } from "./colorUtils";

export default function ButtonGroupComponent({
  uuid,
  value,
  props: { hint, label, visible, disabled, options, colors, sizes },
}: GuiButtonGroupMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  if (!visible) return null;
  return (
    <ViserInputComponent {...{ uuid, hint, label }}>
      {/* Wrapping flex: buttons share each row's width equally (or per
      `sizes`, camtraj patch), but never shrink below their label (minWidth
      fit-content) -- on a narrow panel the overflow wraps onto more rows
      instead of spilling out of the panel. */}
      <Flex wrap="wrap" gap="0.375em">
        {options.map((option, index) => {
          const color = colors?.[index] ?? null;
          const isSelected = option === value;
          return (
            <Button
              key={index}
              onClick={() =>
                messageSender({
                  type: "GuiUpdateMessage",
                  uuid: uuid,
                  updates: { value: option },
                })
              }
              style={{
                flexGrow: sizes?.[index] ?? 1,
                flexBasis: 0,
                minWidth: "fit-content",
              }}
              color={toMantineColor(color)}
              disabled={disabled}
              size="compact-xs"
              variant={color === null ? "outline" : isSelected ? "filled" : "light"}
            >
              {option}
            </Button>
          );
        })}
      </Flex>
    </ViserInputComponent>
  );
}
